#! /usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright 2026 kyh <visionandrobot@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# TimeSurface (~/events_rep) -> RGB sensor_msgs/Image visualizer.
#
# The driver publishes the dense TimeSurface representation at a very high rate
# (hundreds of Hz). Processing every message is wasteful, so this node only
# stores the latest message in the subscription callback and does the (heavier)
# numpy conversion + publish on a fixed-rate timer (default 5 fps).
#
# The RGB mapping is a faithful port of
#   BlinkTrack/util/vis.py::time_surface_to_rgb
# The channel order of the ROS TimeSurface message (channel = 2*bin_idx + p)
# matches BlinkTrack's TimeSurface representation, so the (H, W, C) message data
# is simply transposed to (C, H, W) and fed to the same routine, giving output
# identical to the reference pipeline.

from collections import Counter

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from libcaer_driver_eventframe_msgs.msg import TimeSurface
from sensor_msgs.msg import Image


def time_surface_to_rgb(event_voxel):
    """Faithful port of BlinkTrack/util/vis.py::time_surface_to_rgb.

    event_voxel: float array of shape (C, H, W), channel = 2*bin_idx + polarity.
    Returns an (H, W, 3) uint8 RGB image.
    NOTE: mutates event_voxel in place (first half of channels negated); callers
    should pass a copy.
    """
    C, H, W = event_voxel.shape[-3:]
    img = np.full((H, W, 3), fill_value=255, dtype='uint8')

    event_voxel[:C // 2] = -event_voxel[:C // 2]
    event_image = np.sum(event_voxel, axis=0)

    # assume the most frequent value corresponds to no event
    a_list = event_image.ravel().tolist()
    counts = Counter(a_list)
    most_frequent = counts.most_common(1)
    most_frequent_element = most_frequent[0][0]
    event_image = event_image - most_frequent_element

    max_v = np.max(np.abs(event_image))
    if max_v == 0 or not np.isfinite(max_v):
        event_image = np.zeros_like(event_image, dtype=float)
    else:
        event_image = event_image.astype(float) / float(max_v)

    magnitude = np.abs(event_image) ** 0.25
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    base = 0.2
    color_mag = ((1 - base) * 255 * magnitude)
    color_mag = np.clip(color_mag, 0, 255).astype(np.uint8)
    color_full = np.ones_like(color_mag) * 255
    img[event_image > 0] = np.stack(
        [color_full, 255 - color_mag, 255 - color_mag], axis=-1)[event_image > 0]
    img[event_image < 0] = np.stack(
        [255 - color_mag, 255 - color_mag, color_full], axis=-1)[event_image < 0]

    return img


class TimeSurfaceVisNode(Node):
    def __init__(self):
        super().__init__('timesurface_vis')

        self.input_topic = self.declare_parameter(
            'input_topic', '/event_camera/events_rep').value
        self.output_topic = self.declare_parameter(
            'output_topic', '/event_camera/events_rep_image').value
        self.rate = float(self.declare_parameter('rate', 5.0).value)
        if self.rate <= 0.0:
            self.rate = 5.0

        # match the driver's ~/events_rep QoS (best_effort, volatile, keep last)
        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        self._latest = None       # most recent TimeSurface msg (unprocessed)
        self._last_done = None    # identity of the last processed msg

        self.sub = self.create_subscription(
            TimeSurface, self.input_topic, self._on_msg, qos)
        self.pub = self.create_publisher(Image, self.output_topic, qos)
        self.timer = self.create_timer(1.0 / self.rate, self._on_timer)

        self.get_logger().info(
            f'timesurface_vis: {self.input_topic} -> {self.output_topic} '
            f'@ {self.rate:.1f} fps')

    def _on_msg(self, msg):
        # Cheap: just keep the latest reference; heavy work happens in the timer.
        self._latest = msg

    def _on_timer(self):
        msg = self._latest
        if msg is None or msg is self._last_done:
            return  # nothing new since last publish
        self._last_done = msg

        if msg.height == 0 or msg.width == 0 or msg.channels == 0:
            return
        expected = msg.height * msg.width * msg.channels
        if len(msg.data) != expected:
            self.get_logger().warn(
                f'data size {len(msg.data)} != H*W*C {expected}, skipping')
            return

        # (H, W, C) row-major -> (C, H, W); copy() since the routine mutates it.
        arr = np.asarray(msg.data, dtype=np.float32).reshape(
            msg.height, msg.width, msg.channels)
        voxel = np.transpose(arr, (2, 0, 1)).copy()
        img = time_surface_to_rgb(voxel)  # (H, W, 3) uint8 RGB
        img = np.ascontiguousarray(img)

        out = Image()
        out.header = msg.header  # preserve stamp / frame_id
        out.height = int(msg.height)
        out.width = int(msg.width)
        out.encoding = 'rgb8'
        out.is_bigendian = 0
        out.step = int(msg.width) * 3
        out.data = img.tobytes()
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TimeSurfaceVisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
