// BlinkTrack C++ tracking pipeline — streaming gate loop behind a C-ABI.
// Mirrors scratchpad/c3b/main.cpp exactly, refactored to per-frame streaming.
#include "blinktrack_pipeline.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

torch::Tensor load_bin(const std::string& p, std::vector<int64_t> sh) {
  int64_t n = 1; for (auto s : sh) n *= s;
  std::vector<float> b(n);
  std::ifstream f(p, std::ios::binary);
  f.read((char*)b.data(), n * 4);
  return torch::from_blob(b.data(), sh, torch::kFloat32).clone().to(torch::kCUDA);
}

// C++ port of util/data.py::extract_glimpse (nearest, align_corners, zeros pad).
// SIZE-AGNOSTIC — reads H,W from the actual tensor (the traced glimpse62.ts baked
// the trace-time size and returned zeros at other resolutions). off=(1,2) center
// in pixels (caller passes uc+0.5). Returns (N,C,sz,sz).
torch::Tensor extract_glimpse(torch::Tensor inp, torch::Tensor off, int sz) {
  auto opt = inp.options();
  int64_t W = inp.size(-1), H = inp.size(-2);
  auto ar = torch::arange(0, sz, opt) - (sz - 1) / 2.0;   // xs == ys (square patch)
  auto mg = torch::meshgrid({ar, ar}, "ij");              // {vy, vx}
  auto grid = torch::stack({mg[1], mg[0]}, -1);           // (sz,sz,2) = [vx, vy]
  auto og = off.view({-1, 1, 1, 2}) + grid.unsqueeze(0);  // (N,sz,sz,2)
  auto half = torch::tensor({(double)W / 2.0, (double)H / 2.0}, opt);
  og = (og - half) / half;                                // -> [-1,1]
  namespace F = torch::nn::functional;
  return F::grid_sample(inp, og, F::GridSampleFuncOptions()
      .mode(torch::kNearest).padding_mode(torch::kZeros).align_corners(true));
}

// C++ port of util/data.py::concat_ts (each entry C,H,W) — bit-exact w/ Python.
torch::Tensor concat_ts(const std::vector<torch::Tensor>& L) {
  int64_t n = (int64_t)L.size();
  auto ts = torch::stack(L, -1);
  auto off = torch::linspace(0, 1, n + 1, ts.options()).slice(0, 0, n).view({1, 1, 1, n});
  ts = torch::where(ts > 0, ts / n + off, torch::zeros_like(ts));
  int64_t C = ts.size(0), H = ts.size(1), W = ts.size(2);
  ts = ts.reshape({C / 2, 2, H, W, n}).permute({2, 3, 1, 4, 0}).flatten(3, 4);
  return ts.reshape({H, W, 2, C / 2, -1}).amax(4).permute({0, 1, 3, 2}).flatten(2, 3).permute({2, 0, 1});
}

}  // namespace

struct BTPipeline {
  torch::jit::Module te, rest, policy, refinit;  // glimpse now done in C++ (extract_glimpse)
  bool has_refinit = false;
  torch::Tensor dref, frm, frs;
  torch::Tensor uc, h, c, pxr, feat, vec;
  std::vector<torch::Tensor> stack;
  bool started = false;
  float init_x_, init_y_;
  int max_stack_ = 64;    // ring-buffer cap: drop oldest patch once list exceeds N
  int accum_count_ = 0;   // TRUE accumulation count since last fire (uncapped)
  bool use_rl_ = true;    // false: skip RL gate, run the tracker every frame

  // --- RL I/O capture (debug logging; gated to avoid per-frame GPU->CPU copy) ---
  bool capture_ = false;
  std::vector<float> last_feat_ = std::vector<float>(10 * 62 * 62, 0.0f);  // RL input feature_map
  float last_slen_ = 0, last_logits_[2] = {0, 0}, last_kp_[2] = {0, 0};
  int last_action_ = 0;

  BTPipeline(const std::string& d, float ix, float iy) : init_x_(ix), init_y_(iy) {
    te     = torch::jit::load(d + "/te.ts");           te.to(torch::kCUDA);     te.eval();
    rest   = torch::jit::load(d + "/tracker_rest.ts");  rest.to(torch::kCUDA);   rest.eval();
    policy = torch::jit::load(d + "/policy.ts");        policy.to(torch::kCUDA); policy.eval();
    dref = load_bin(d + "/d_ref.bin", {1, 384, 1, 1});
    frm  = load_bin(d + "/f_ref_mid.bin", {1, 128, 31, 31});
    frs  = load_bin(d + "/f_ref_squ.bin", {1, 128, 31, 31});
    // optional: on-camera reference init from a grayscale patch (register_ref.ts)
    try {
      refinit = torch::jit::load(d + "/register_ref.ts"); refinit.to(torch::kCUDA);
      refinit.eval(); has_refinit = true;
    } catch (...) { has_refinit = false; }
    reset_state();
    warmup();  // trigger CUDA/TRT/grid_sample lazy init so the first real frame is fast
  }

  // Compute the reference (d_ref/f_ref_mid/f_ref_squ) from a grayscale frame
  // (H,W in [0,1]) around center (cx,cy) — replaces the baked reference. Also
  // re-inits the keypoint to (cx,cy). Returns false if register_ref.ts absent.
  bool register_ref(const float* gray, int H, int W, float cx, float cy) {
    if (!has_refinit) return false;
    torch::NoGradGuard ng;
    auto g = torch::from_blob((void*)gray, {1, 1, H, W}, torch::kFloat32).clone().to(torch::kCUDA);
    auto ctr = torch::tensor({{cx, cy}},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) + 0.5f;
    auto patch = extract_glimpse(g, ctr, 62);   // (1,1,62,62) via SIZE-AGNOSTIC C++ glimpse
    auto out = refinit.forward({patch}).toTuple();  // register_ref.ts takes the 62x62 patch
    dref = out->elements()[0].toTensor();
    frm  = out->elements()[1].toTensor();
    frs  = out->elements()[2].toTensor();
    init_x_ = cx; init_y_ = cy;
    reset_state();
    return true;
  }

  void reset_state() {
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    uc  = torch::tensor({{init_x_, init_y_}}, opt);
    h   = torch::zeros({1, 128, 7, 7}, opt);
    c   = torch::zeros({1, 128, 7, 7}, opt);
    pxr = torch::zeros({1, 256}, opt);
    stack.clear();
    accum_count_ = 0;
    started = false;
  }

  // Run the FULL process() path (crop+concat+policy+te+rest) on dummy frames so
  // every kernel + the CUDA context are hot; then restore the initial state. The
  // first real frame would otherwise take ~1s (lazy init) and, with a live
  // publisher, cause dropped frames that diverge the (chaotic) track.
  void warmup() {
    const int H = 180, W = 240, C = 10;  // CUDA-context warmup is size-independent
    std::vector<float> dummy((size_t)H * W * C, 0.1f);
    float xy[2];
    for (int i = 0; i < 4; i++) process(dummy.data(), H, W, C, xy);
    reset_state();
  }

  // frame_hwc: (H,W,C) row-major. Crop a 62x62 patch around uc, return its (C,62,62).
  torch::Tensor crop(const float* frame_hwc, int H, int W, int C) {
    auto host = torch::from_blob((void*)frame_hwc, {H, W, C}, torch::kFloat32);
    auto inp = host.permute({2, 0, 1}).contiguous().unsqueeze(0).to(torch::kCUDA);  // 1,C,H,W
    return extract_glimpse(inp, uc + 0.5f, 62)[0];                                   // C,62,62
  }

  // Append a patch; if the list exceeds the cap, drop the oldest (ring buffer)
  // to bound concat_ts cost. accum_count_ tracks the TRUE (uncapped) count so
  // the RL's stacked_length keeps growing even though the list is bounded — this
  // preserves the RL's "fire when accumulated too long" behavior (a capped
  // stacked_length would saturate and freeze the tracker).
  void push_capped(torch::Tensor p) {
    stack.push_back(std::move(p));
    if ((int)stack.size() > max_stack_) stack.erase(stack.begin());
    accum_count_++;
  }

  // Cache the RL input (feature_map=feat, stacked_length=vec) + output (action,
  // logits) + current keypoint for debug logging. Called only when capture_ is on
  // (a GPU->CPU copy of the 10x62x62 feature per frame, so gated off by default).
  void capture_rl_io(int action, float logit0, float logit1) {
    auto fcpu = feat.to(torch::kCPU).contiguous();     // (1,10,62,62)
    std::memcpy(last_feat_.data(), fcpu.data_ptr<float>(), last_feat_.size() * sizeof(float));
    last_slen_ = vec.to(torch::kCPU)[0][0].item<float>();
    last_action_ = action;
    last_logits_[0] = logit0; last_logits_[1] = logit1;
    auto ucc = uc.to(torch::kCPU);
    last_kp_[0] = ucc[0][0].item<float>(); last_kp_[1] = ucc[0][1].item<float>();
  }

  int process(const float* frame_hwc, int H, int W, int C, float* out_xy) {
    torch::NoGradGuard ng;
    int fired = 0;
    if (!started) {
      started = true;
      stack.clear();
      accum_count_ = 0;
      push_capped(crop(frame_hwc, H, W, C));
      feat = concat_ts(stack).unsqueeze(0);
      vec = torch::tensor({{(float)accum_count_}}, uc.options());
    } else {
      // use_rl_ off: never accumulate, run the tracker (fire) every frame.
      // policy.ts returns a tuple (action[int64,B], logits[float,B,2]).
      int action = 1;
      float logit0 = 0.0f, logit1 = 0.0f;
      if (use_rl_) {
        auto pol = policy.forward({feat, vec}).toTuple();
        action = (int)pol->elements()[0].toTensor()[0].item<int64_t>();
        auto lg = pol->elements()[1].toTensor()[0];   // shape [2]
        logit0 = lg[0].item<float>();
        logit1 = lg[1].item<float>();
      }
      if (capture_) capture_rl_io(action, logit0, logit1);  // feat/vec (RL input) + action/logits
      if (action == 1) {
        auto f0 = te.forward({feat}).toTensor();
        auto out = rest.forward({f0, dref, frm, frs, h, c, pxr}).toTuple();
        auto coord = out->elements()[0].toTensor();
        h = out->elements()[1].toTensor();
        c = out->elements()[2].toTensor();
        pxr = out->elements()[3].toTensor();
        uc = uc + coord;
        stack.clear();
        accum_count_ = 0;
        fired = 1;
      }
      push_capped(crop(frame_hwc, H, W, C));
      feat = concat_ts(stack).unsqueeze(0);
      vec = torch::tensor({{(float)accum_count_}}, uc.options());
    }
    auto ucc = uc.to(torch::kCPU);
    out_xy[0] = ucc[0][0].item<float>();
    out_xy[1] = ucc[0][1].item<float>();
    return fired;
  }
};

extern "C" {

BTPipeline* bt_create(const char* model_dir, float init_x, float init_y) {
  try {
    return new BTPipeline(std::string(model_dir), init_x, init_y);
  } catch (const std::exception& e) {
    fprintf(stderr, "[bt_create] %s\n", e.what());
    return nullptr;
  }
}

void bt_set_max_accum(BTPipeline* p, int n) {
  if (p && n > 0) p->max_stack_ = n;
}

void bt_set_use_rl(BTPipeline* p, int on) {
  if (p) p->use_rl_ = (on != 0);
}

void bt_set_capture(BTPipeline* p, int on) {
  if (p) p->capture_ = (on != 0);
}

void bt_get_rl_io(BTPipeline* p, float* feat_out, float* slen_out,
                  int* action_out, float* logits_out, float* kp_out) {
  if (!p) return;
  if (feat_out) std::memcpy(feat_out, p->last_feat_.data(), p->last_feat_.size() * sizeof(float));
  if (slen_out) *slen_out = p->last_slen_;
  if (action_out) *action_out = p->last_action_;
  if (logits_out) { logits_out[0] = p->last_logits_[0]; logits_out[1] = p->last_logits_[1]; }
  if (kp_out) { kp_out[0] = p->last_kp_[0]; kp_out[1] = p->last_kp_[1]; }
}

int bt_register_ref(BTPipeline* p, const float* gray, int H, int W, float cx, float cy) {
  if (!p) return -1;
  try {
    return p->register_ref(gray, H, W, cx, cy) ? 0 : 1;  // 0 ok, 1 no register_ref.ts
  } catch (const std::exception& e) {
    fprintf(stderr, "[bt_register_ref] %s\n", e.what());
    return -1;
  }
}

int bt_process(BTPipeline* p, const float* frame_hwc, int H, int W, int C, float* out_xy) {
  if (!p) return -1;
  try {
    return p->process(frame_hwc, H, W, C, out_xy);
  } catch (const std::exception& e) {
    fprintf(stderr, "[bt_process] %s\n", e.what());
    return -1;
  }
}

void bt_destroy(BTPipeline* p) { delete p; }

}  // extern "C"
