#! /usr/bin/env python3
# -----------------------------------------------------------------------------
# blinktrack_node: subscribes to /events_rep (TimeSurface event frames) and runs
# the RL-gated BlinkTrack tracker frame-by-frame, publishing tracked keypoints.
#
# Uses BlinkTrackStreamer (blinktrack_core), which is bit-identical to the
# offline BlinkTrack pipeline. The rclpy callback only enqueues frames; the
# heavy tracker+RL step runs in a worker thread paced by frame arrival.
#
# Run in the BlinkTrack env (gostop) with ROS sourced + LD_PRELOAD +
# PYTHONNOUSERSITE=1 + BlinkTrack on PYTHONPATH (see project memory).
# -----------------------------------------------------------------------------
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from libcaer_driver_eventframe_msgs.msg import TimeSurface
from geometry_msgs.msg import PointStamped

from blinktrack_core import BlinkTrackStreamer


class BlinkTrackNode(Node):
    def __init__(self):
        super().__init__("blinktrack_node")
        in_topic = self.declare_parameter("input_topic", "/event_camera/events_rep").value
        out_topic = self.declare_parameter("output_topic", "/event_camera/track").value
        kx = float(self.declare_parameter("init_x", 126.0).value)
        ky = float(self.declare_parameter("init_y", 37.0).value)
        self.max_steps = int(self.declare_parameter("max_steps", 0).value)  # 0 = run forever
        self.dump_path = self.declare_parameter("dump_path", "").value
        # tracker needs every frame in order; RELIABLE for lossless replay/validation.
        # For the real driver (~/events_rep is BEST_EFFORT) leave False and accept drops.
        reliable = bool(self.declare_parameter("reliable", False).value)

        self.get_logger().info(f"loading BlinkTrackStreamer (init kp=({kx},{ky})) ...")
        self.streamer = BlinkTrackStreamer(init_keypoints=[[kx, ky]])
        self.get_logger().info("streamer ready")

        qos = QoSProfile(depth=2000)
        qos.reliability = ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        self.sub = self.create_subscription(TimeSurface, in_topic, self._on_frame, qos)
        self.pub = self.create_publisher(PointStamped, out_topic, 10)

        self._n_rx = 0
        self._latencies = []
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()
        self.get_logger().info(f"subscribed {in_topic} -> publishing {out_topic}")

    def _on_frame(self, msg):
        hwc = np.asarray(msg.data, dtype=np.float32).reshape(msg.height, msg.width, msg.channels)
        self.streamer.push(hwc)
        self._n_rx += 1
        if self._n_rx == 1:
            self.get_logger().info(f"first frame received ({msg.height}x{msg.width}x{msg.channels})")

    def _run_loop(self):
        self.get_logger().info("worker: waiting for first frame to reset()...")
        self.streamer.reset()  # blocks for the first frame
        self.get_logger().info("worker: reset done, tracking loop started")
        i = 0
        while rclpy.ok():
            t0 = time.perf_counter()
            kp, action = self.streamer.step()  # blocks for the next frame
            self._latencies.append(time.perf_counter() - t0)
            ps = PointStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = "camera"
            ps.point.x = float(kp[0, 0])
            ps.point.y = float(kp[0, 1])
            ps.point.z = float(action[0])  # z carries the fire/accumulate flag
            self.pub.publish(ps)
            i += 1
            if i % 100 == 0:
                self.get_logger().info(f"worker: step {i}, kp=({kp[0,0]:.1f},{kp[0,1]:.1f})")
            if self.max_steps and i >= self.max_steps:
                break
        self._finish(i)

    def _finish(self, n):
        lat = np.array(self._latencies[1:]) * 1e3  # drop warm-up step
        self.get_logger().info(
            f"done: {n} steps, step latency mean={lat.mean():.2f}ms p50={np.percentile(lat,50):.2f} "
            f"p95={np.percentile(lat,95):.2f}ms")
        if self.dump_path:
            np.savetxt(self.dump_path, self.streamer.track_data,
                       fmt=["%i", "%.9f", "%.6f", "%.6f"])
            self.get_logger().info(f"saved track -> {self.dump_path}")


def main(args=None):
    rclpy.init(args=args)
    node = BlinkTrackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
