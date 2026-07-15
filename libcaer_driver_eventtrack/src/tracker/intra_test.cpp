// Intra-process composable test: publisher node + tracker node in ONE process,
// ONE executor, use_intra_process_comms(true) => zero-copy TimeSurface handoff.
// This is exactly the "same ComposableNodeContainer as the driver" scenario.
// Publishes frames.bin as TimeSurface; tracker runs the dlopen'd torch pipeline;
// measures per-frame end-to-end latency (publish -> callback done); then exits.
#include "blinktrack_pipeline.h"
#include <geometry_msgs/msg/point_stamped.hpp>
#include <libcaer_driver_eventframe_msgs/msg/time_surface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <memory>
#include <vector>

using libcaer_driver_eventframe_msgs::msg::TimeSurface;
using Clock = std::chrono::high_resolution_clock;

// dlopen the ABI=0 torch pipeline in an isolated DEEPBIND|LOCAL scope.
struct Lib {
  BTPipeline* (*create)(const char*, float, float);
  int (*process)(BTPipeline*, const float*, int, int, int, float*);
  void (*destroy)(BTPipeline*);
  Lib(const char* so) {
    void* h = dlopen(so, RTLD_NOW | RTLD_DEEPBIND | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); std::abort(); }
    create  = (decltype(create))dlsym(h, "bt_create");
    process = (decltype(process))dlsym(h, "bt_process");
    destroy = (decltype(destroy))dlsym(h, "bt_destroy");
  }
};

int main(int argc, char** argv) {
  std::string c3b = argv[1], models = argv[2], solib = argv[3];
  double hz = argc > 4 ? std::stod(argv[4]) : 200.0;

  std::map<std::string, int> meta;
  { std::ifstream mf(c3b + "/meta.txt"); std::string k; int v; while (mf >> k >> v) meta[k] = v; }
  int nf = meta["nframes"], C = meta["C"], H = meta["H"], W = meta["W"];
  float ix = meta["init_x"], iy = meta["init_y"];
  int64_t fsz = (int64_t)C * H * W;
  std::vector<float> all((int64_t)nf * fsz);
  { std::ifstream f(c3b + "/frames.bin", std::ios::binary); f.read((char*)all.data(), all.size() * 4); }
  // pre-convert C,H,W -> H,W,C (msg layout)
  std::vector<std::vector<float>> hwc(nf, std::vector<float>(fsz));
  for (int fi = 0; fi < nf; fi++) {
    const float* chw = all.data() + (int64_t)fi * fsz;
    for (int cc = 0; cc < C; cc++)
      for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
          hwc[fi][(y * W + x) * C + cc] = chw[(cc * H + y) * W + x];
  }

  Lib lib(solib.c_str());
  BTPipeline* pipe = lib.create(models.c_str(), ix, iy);
  if (!pipe) { fprintf(stderr, "bt_create failed\n"); return 1; }

  rclcpp::init(argc, argv);
  auto ctx_opts = rclcpp::NodeOptions().use_intra_process_comms(true);
  auto pub_node = std::make_shared<rclcpp::Node>("mock_pub", ctx_opts);
  auto trk_node = std::make_shared<rclcpp::Node>("tracker", ctx_opts);
  auto pub = pub_node->create_publisher<TimeSurface>("/events_rep", rclcpp::SensorDataQoS());
  auto trk_pub = trk_node->create_publisher<geometry_msgs::msg::PointStamped>("/track", 10);

  std::vector<double> lat;  // per-frame end-to-end ms
  std::vector<std::pair<float,float>> track = {{ix, iy}};
  int recv = 0, fires = 0;
  std::vector<Clock::time_point> pub_ts(nf);

  auto sub = trk_node->create_subscription<TimeSurface>(
    "/events_rep", rclcpp::SensorDataQoS(),
    [&](const TimeSurface::ConstSharedPtr msg) {
      int idx = recv;  // frames arrive in order
      auto t0 = Clock::now();
      float xy[2];
      int fired = lib.process(pipe, msg->data.data(), (int)msg->height, (int)msg->width,
                              (int)msg->channels, xy);
      auto t1 = Clock::now();
      double e2e = std::chrono::duration<double, std::milli>(t1 - pub_ts[idx]).count();
      lat.push_back(e2e);
      if (fired > 0) { fires++; track.push_back({xy[0], xy[1]}); }
      geometry_msgs::msg::PointStamped p; p.header = msg->header;
      p.point.x = xy[0]; p.point.y = xy[1]; p.point.z = fired ? 1.0 : 0.0;
      trk_pub->publish(p);
      recv++;
      (void)idx; (void)t0;
    });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(pub_node);
  exec.add_node(trk_node);

  int sent = 0;
  auto period = std::chrono::duration<double>(1.0 / hz);
  auto timer = pub_node->create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    [&]() {
      if (sent >= nf) { if (recv >= nf) rclcpp::shutdown(); return; }
      auto msg = std::make_unique<TimeSurface>();
      msg->header.stamp = pub_node->now();
      msg->height = H; msg->width = W; msg->channels = C; msg->n_bins = C / 2;
      msg->data = hwc[sent];              // copy into msg (unavoidable: msg owns data)
      pub_ts[sent] = Clock::now();
      pub->publish(std::move(msg));       // intra-process: subscriber gets the same buffer (zero-copy)
      sent++;
    });

  auto t_start = Clock::now();
  exec.spin();
  double wall = std::chrono::duration<double>(Clock::now() - t_start).count();
  lib.destroy(pipe);

  std::sort(lat.begin(), lat.end());
  double sum = 0; for (double v : lat) sum += v;
  auto pct = [&](double p){ return lat.empty()?0.0:lat[std::min(lat.size()-1,(size_t)(p*lat.size()))]; };
  printf("=== intra-process composable end-to-end ===\n");
  printf("frames sent=%d recv=%d fires=%d | track rows=%zu last=(%.2f,%.2f)\n",
         sent, recv, fires, track.size(), track.back().first, track.back().second);
  printf("e2e latency ms: mean=%.3f  p50=%.3f  p95=%.3f  p99=%.3f  max=%.3f\n",
         lat.empty()?0:sum/lat.size(), pct(0.50), pct(0.95), pct(0.99), pct(0.999));
  printf("wall=%.3fs  throughput=%.1f frames/s\n", wall, recv / wall);
  return 0;
}
