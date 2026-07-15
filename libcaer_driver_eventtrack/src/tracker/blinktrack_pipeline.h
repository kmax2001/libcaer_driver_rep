// C-ABI boundary for the BlinkTrack C++ tracking pipeline.
// The implementation (.cpp) links libtorch + torch_tensorrt and is compiled
// with _GLIBCXX_USE_CXX11_ABI=0 (torch's ABI). This header exposes ONLY plain
// C types so an ABI=1 consumer (e.g. an rclcpp node) can call it without an
// ABI mismatch — no std::string / std::vector crosses the boundary.
#ifdef __cplusplus
extern "C" {
#endif

typedef struct BTPipeline BTPipeline;

// Create a pipeline. model_dir must contain:
//   te.ts, tracker_rest.ts, policy.ts, glimpse62.ts,
//   d_ref.bin (1,384,1,1), f_ref_mid.bin (1,128,31,31), f_ref_squ.bin (1,128,31,31)
// init_x/init_y: initial keypoint (pixels). Returns NULL on failure.
BTPipeline* bt_create(const char* model_dir, float init_x, float init_y);

// Ring-buffer accumulation cap: drop the oldest patch once accumulation exceeds
// n (bounds concat_ts latency). Default 64.
void bt_set_max_accum(BTPipeline* p, int n);

// Enable/disable the RL gate. on=0: skip the policy and run the tracker every
// frame (no accumulation gating). on!=0 (default): RL decides fire/accumulate.
void bt_set_use_rl(BTPipeline* p, int on);

// Enable/disable per-frame caching of the RL input/output for debug logging
// (a 10x62x62 GPU->CPU copy per frame while on; off by default).
void bt_set_capture(BTPipeline* p, int on);

// Copy the last frame's cached RL I/O (call after bt_process, capture on).
//   feat_out   : feature_map, 10*62*62 floats (the RL input time surface)
//   slen_out   : stacked_length (1 float)
//   action_out : 0/1 (1 int)
//   logits_out : pre-argmax scores [accumulate, fire] (2 floats)
//   kp_out     : keypoint x,y (2 floats)
// Any pointer may be null to skip that field.
void bt_get_rl_io(BTPipeline* p, float* feat_out, float* slen_out,
                  int* action_out, float* logits_out, float* kp_out);

// Compute the reference from a grayscale frame (H,W row-major, values in [0,1])
// around center (cx,cy) — a 62x62 square patch — replacing the baked reference
// and re-initializing the keypoint to (cx,cy). Returns 0 on success, 1 if the
// model dir has no register_ref.ts, -1 on error.
int bt_register_ref(BTPipeline* p, const float* gray, int H, int W, float cx, float cy);

// Process one TimeSurface frame in (H, W, C) row-major layout (msg->data).
// Writes the current keypoint (x, y) to out_xy[2]. Returns 1 if the tracker
// fired on this frame, 0 otherwise, -1 on error.
int bt_process(BTPipeline* p, const float* frame_hwc, int H, int W, int C, float* out_xy);

void bt_destroy(BTPipeline* p);

#ifdef __cplusplus
}
#endif
