// rclcpp tracker node (compiled ABI=1). Subscribes /events_rep (TimeSurface),
// runs the ABI=0 BlinkTrack pipeline via its C API, publishes the keypoint.
// NO torch symbols cross into this TU — only plain C types from the header.
//
// The pipeline lib is dlopen'd with RTLD_DEEPBIND so torch's internal ABI=0
// std::regex/std::string template instantiations bind to torch's own copies
// instead of being interposed by rclcpp's ABI=1 copies (which corrupts the
// heap inside torch::jit::load). This is the key to co-hosting torch + rclcpp.
#include "blinktrack_pipeline.h"
#include <geometry_msgs/msg/point_stamped.hpp>
#include <libcaer_driver_eventframe_msgs/msg/time_surface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/empty.hpp>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

// C-API function pointers resolved from the dlopen'd pipeline lib.
namespace btlib {
using create_t   = BTPipeline* (*)(const char*, float, float);
using process_t  = int (*)(BTPipeline*, const float*, int, int, int, float*);
using destroy_t  = void (*)(BTPipeline*);
using setmax_t   = void (*)(BTPipeline*, int);
using userl_t    = void (*)(BTPipeline*, int);
using regref_t   = int (*)(BTPipeline*, const float*, int, int, float, float);
using capture_t  = void (*)(BTPipeline*, int);
using getio_t    = void (*)(BTPipeline*, float*, float*, int*, float*, float*);
static create_t   bt_create      = nullptr;
static process_t  bt_process     = nullptr;
static destroy_t  bt_destroy     = nullptr;
static setmax_t   bt_set_max_accum = nullptr;
static userl_t    bt_set_use_rl    = nullptr;
static regref_t   bt_register_ref  = nullptr;
static capture_t  bt_set_capture   = nullptr;
static getio_t    bt_get_rl_io     = nullptr;

static bool load(const std::string& so_path, std::string& err) {
  // LOCAL (not GLOBAL) + DEEPBIND: torch loads inside this lib's own scope and
  // resolves its ABI=0 symbols to itself, invisible to rclcpp's ABI=1 globals.
  void* h = dlopen(so_path.c_str(), RTLD_NOW | RTLD_DEEPBIND | RTLD_LOCAL);
  if (!h) { err = dlerror() ? dlerror() : "dlopen failed"; return false; }
  bt_create  = (create_t)dlsym(h, "bt_create");
  bt_process = (process_t)dlsym(h, "bt_process");
  bt_destroy = (destroy_t)dlsym(h, "bt_destroy");
  bt_set_max_accum = (setmax_t)dlsym(h, "bt_set_max_accum");
  bt_set_use_rl    = (userl_t)dlsym(h, "bt_set_use_rl");
  bt_register_ref  = (regref_t)dlsym(h, "bt_register_ref");
  bt_set_capture   = (capture_t)dlsym(h, "bt_set_capture");
  bt_get_rl_io     = (getio_t)dlsym(h, "bt_get_rl_io");
  if (!bt_create || !bt_process || !bt_destroy) { err = "dlsym failed"; return false; }
  return true;
}
}  // namespace btlib

using libcaer_driver_eventframe_msgs::msg::TimeSurface;

class BlinkTrackTrackerNode : public rclcpp::Node {
public:
  explicit BlinkTrackTrackerNode(const rclcpp::NodeOptions& opts)
    : rclcpp::Node("blinktrack_tracker", opts) {
    auto lib = declare_parameter<std::string>("pipeline_lib", "libblinktrack_pipeline.so");
    auto model_dir = declare_parameter<std::string>("model_dir", "");
    // init keypoint: <0 means "use the image center" (resolved on first image).
    init_x_ = declare_parameter<double>("init_x", -1.0);
    init_y_ = declare_parameter<double>("init_y", -1.0);
    int max_accum = declare_parameter<int>("max_accum", 64);
    bool use_rl = declare_parameter<bool>("use_rl", true);
    // register the reference from the first /image_raw grayscale frame (real
    // camera). false = use the baked reference immediately (offline/EC testing).
    bool cam_ref = declare_parameter<bool>("use_camera_reference", true);
    // RL I/O debug logging (triggered by ~/log_trigger).
    log_dir_ = declare_parameter<std::string>("log_dir", "/tmp/blinktrack_rllog");
    log_seconds_ = declare_parameter<double>("log_seconds", 0.5);
    std::string err;
    if (!btlib::load(lib, err)) {
      RCLCPP_FATAL(get_logger(), "load pipeline lib '%s' failed: %s", lib.c_str(), err.c_str());
      throw std::runtime_error("pipeline lib load failed");
    }
    pipe_ = btlib::bt_create(model_dir.c_str(), (float)init_x_, (float)init_y_);
    if (!pipe_) {
      RCLCPP_FATAL(get_logger(), "bt_create failed (model_dir=%s)", model_dir.c_str());
      throw std::runtime_error("bt_create failed");
    }
    if (btlib::bt_set_max_accum) btlib::bt_set_max_accum(pipe_, max_accum);
    if (btlib::bt_set_use_rl) btlib::bt_set_use_rl(pipe_, use_rl ? 1 : 0);
    pub_ = create_publisher<geometry_msgs::msg::PointStamped>("~/track", 10);
    sub_ = create_subscription<TimeSurface>(
      "/events_rep", rclcpp::SensorDataQoS(),
      [this](const TimeSurface::ConstSharedPtr msg) { onFrame(msg); });
    // RL I/O logging trigger (from the OpenCV 'd' key). Reliable (must not drop).
    log_sub_ = create_subscription<std_msgs::msg::Empty>(
      "~/log_trigger", 10,
      [this](const std_msgs::msg::Empty::ConstSharedPtr) { onLogTrigger(); });

    ref_ready_ = !cam_ref;  // baked mode: ready now; camera mode: wait for image
    if (cam_ref) {
      if (!btlib::bt_register_ref) {
        RCLCPP_FATAL(get_logger(), "use_camera_reference but lib has no bt_register_ref");
        throw std::runtime_error("no bt_register_ref");
      }
      img_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/image_raw", rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) { onImage(msg); });
      // external UI (OpenCV click node) can re-center the tracker at any time
      reinit_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
        "~/reinit_point", 10,
        [this](const geometry_msgs::msg::PointStamped::ConstSharedPtr pt) { onReinit(pt); });
      RCLCPP_INFO(get_logger(), "blinktrack_tracker up: waiting for /image_raw to register reference"
                                " (center=%s); ~/reinit_point re-centers on click",
                  init_x_ >= 0 ? "param" : "image-center");
    } else {
      RCLCPP_INFO(get_logger(), "blinktrack_tracker up: baked reference, init=(%.1f,%.1f)",
                  init_x_, init_y_);
    }
  }
  ~BlinkTrackTrackerNode() override { if (pipe_) btlib::bt_destroy(pipe_); }

private:
  // Convert an /image_raw msg to grayscale [0,1] and (re)register the reference
  // at (cx,cy). bt_register_ref also resets the LSTM state + accumulation stack.
  bool registerRefAt(const sensor_msgs::msg::Image::ConstSharedPtr& msg, float cx, float cy) {
    const int H = (int)msg->height, W = (int)msg->width;
    const auto& d = msg->data;
    const int step = (int)msg->step;
    const std::string& enc = msg->encoding;
    auto u16 = [&](int off) { return (uint16_t)(d[off] | (d[off + 1] << 8)); };  // little-endian
    std::vector<float> gray((size_t)H * W);
    if (enc == "mono8") {
      for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
        gray[(size_t)y * W + x] = d[(size_t)y * step + x] / 255.0f;
    } else if (enc == "mono16") {
      for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
        gray[(size_t)y * W + x] = u16((int)((size_t)y * step + 2 * x)) / 65535.0f;
    } else if (enc == "rgb16" || enc == "bgr16") {
      for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int o = (int)((size_t)y * step + 6 * x);
        gray[(size_t)y * W + x] = (u16(o) + u16(o + 2) + u16(o + 4)) / 3.0f / 65535.0f;
      }
    } else if (enc == "rgb8" || enc == "bgr8") {
      for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int o = (int)((size_t)y * step + 3 * x);
        gray[(size_t)y * W + x] = (d[o] + d[o + 1] + d[o + 2]) / 3.0f / 255.0f;
      }
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "unsupported /image_raw encoding '%s' (%dx%d)", enc.c_str(), H, W);
      return false;
    }
    if (cx < 0 || cy < 0 || cx >= W || cy >= H) {
      RCLCPP_WARN(get_logger(), "reinit point (%.1f,%.1f) out of %dx%d image", cx, cy, W, H);
      return false;
    }
    int rc = btlib::bt_register_ref(pipe_, gray.data(), H, W, cx, cy);
    if (rc != 0) { RCLCPP_WARN(get_logger(), "bt_register_ref rc=%d", rc); return false; }
    return true;
  }

  // Keep the latest grayscale frame; auto-register at center on the first one.
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    latest_img_ = msg;
    if (ref_ready_) return;
    float cx = init_x_ >= 0 ? (float)init_x_ : msg->width / 2.0f;
    float cy = init_y_ >= 0 ? (float)init_y_ : msg->height / 2.0f;
    if (registerRefAt(msg, cx, cy)) {
      ref_ready_ = true;
      RCLCPP_INFO(get_logger(), "reference registered at (%.1f,%.1f) from %ux%u image; tracking",
                  cx, cy, msg->width, msg->height);
    }
  }

  // Re-initialize the tracker at a clicked point (from an external UI node):
  // re-register the reference there + reset LSTM state + reset accumulation.
  void onReinit(const geometry_msgs::msg::PointStamped::ConstSharedPtr& pt) {
    if (!latest_img_) {
      RCLCPP_WARN(get_logger(), "reinit requested but no /image_raw frame yet");
      return;
    }
    if (registerRefAt(latest_img_, (float)pt->point.x, (float)pt->point.y)) {
      ref_ready_ = true;
      RCLCPP_INFO(get_logger(), "re-initialized at click (%.1f,%.1f)", pt->point.x, pt->point.y);
    }
  }

  // 'd' key (from the OpenCV node) -> capture RL I/O for the next log_seconds.
  void onLogTrigger() {
    if (!btlib::bt_set_capture || !btlib::bt_get_rl_io) {
      RCLCPP_WARN(get_logger(), "log requested but lib has no RL-I/O capture API");
      return;
    }
    log_capdir_ = log_dir_ + "/capture_" + std::to_string(now().nanoseconds());
    std::error_code ec;
    std::filesystem::create_directories(log_capdir_, ec);
    if (ec) { RCLCPP_WARN(get_logger(), "mkdir %s failed", log_capdir_.c_str()); return; }
    { std::ofstream m(log_capdir_ + "/meta.txt");
      m << "feat_shape 10 62 62\n"
           "fields feat[38440] stacked_length action logit_accumulate logit_fire kp_x kp_y\n"
           "dtype float32\n"; }
    btlib::bt_set_capture(pipe_, 1);
    logging_ = true; log_t0_ = now(); log_idx_ = 0;
    RCLCPP_INFO(get_logger(), "RL-I/O logging started -> %s", log_capdir_.c_str());
  }

  void writeLogFrame() {
    static thread_local std::vector<float> feat(10 * 62 * 62);
    float slen, logits[2], kp[2]; int action;
    btlib::bt_get_rl_io(pipe_, feat.data(), &slen, &action, logits, kp);
    char name[64]; std::snprintf(name, sizeof(name), "/frame_%05d.bin", log_idx_);
    std::ofstream f(log_capdir_ + name, std::ios::binary);
    f.write((const char*)feat.data(), feat.size() * sizeof(float));
    float a = (float)action;
    f.write((const char*)&slen, 4); f.write((const char*)&a, 4);
    f.write((const char*)logits, 8); f.write((const char*)kp, 8);
    log_idx_++;
  }

  void onFrame(const TimeSurface::ConstSharedPtr& msg) {
    if (!ref_ready_) return;  // wait until the reference patch is registered
    const int H = (int)msg->height, W = (int)msg->width, C = (int)msg->channels;
    if ((int64_t)msg->data.size() != (int64_t)H * W * C) {
      RCLCPP_WARN(get_logger(), "bad data size %zu != %d*%d*%d", msg->data.size(), H, W, C);
      return;
    }
    float xy[2];
    auto t0 = std::chrono::high_resolution_clock::now();
    int fired = btlib::bt_process(pipe_, msg->data.data(), H, W, C, xy);
    double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - t0).count();
    if (fired < 0) return;

    // RL I/O logging window
    if (logging_) {
      if ((now() - log_t0_).seconds() < log_seconds_) {
        writeLogFrame();
      } else {
        logging_ = false;
        btlib::bt_set_capture(pipe_, 0);
        RCLCPP_INFO(get_logger(), "RL-I/O logging done: saved %d frames to %s",
                    log_idx_, log_capdir_.c_str());
      }
    }

    geometry_msgs::msg::PointStamped p;
    p.header = msg->header;
    p.point.x = xy[0]; p.point.y = xy[1]; p.point.z = fired ? 1.0 : 0.0;
    pub_->publish(p);

    // latency stats
    n_++; sum_ms_ += ms; if (ms > max_ms_) max_ms_ = ms;
    if (n_ % 50 == 0)
      RCLCPP_INFO(get_logger(), "n=%ld kp=(%.1f,%.1f) fired=%d | proc %.2fms (avg %.2f, max %.2f)",
                  n_, xy[0], xy[1], fired, ms, sum_ms_ / n_, max_ms_);
  }

  BTPipeline* pipe_ = nullptr;
  double init_x_, init_y_;
  bool ref_ready_ = false;
  rclcpp::Subscription<TimeSurface>::SharedPtr sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr reinit_sub_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_img_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr pub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr log_sub_;
  long n_ = 0; double sum_ms_ = 0, max_ms_ = 0;
  // RL I/O logging state
  std::string log_dir_, log_capdir_;
  double log_seconds_ = 0.5;
  bool logging_ = false;
  rclcpp::Time log_t0_;
  int log_idx_ = 0;
};

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(BlinkTrackTrackerNode)

#ifndef BLINKTRACK_COMPONENT_ONLY
// Also runnable as a standalone executable.
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opts;
  opts.use_intra_process_comms(true);
  rclcpp::spin(std::make_shared<BlinkTrackTrackerNode>(opts));
  rclcpp::shutdown();
  return 0;
}
#endif
