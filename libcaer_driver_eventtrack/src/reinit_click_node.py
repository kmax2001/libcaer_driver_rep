#! /usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright 2026 kyh <visionandrobot@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0
# -----------------------------------------------------------------------------
# OpenCV click-to-reinit UI for the BlinkTrack tracker.
#
# Shows the event-surface RGB image (from timesurface_vis_node, which already
# overlays the current keypoint) in an OpenCV window. A left mouse click at
# (x, y) publishes a PointStamped on ~/reinit_point which the tracker node uses
# to re-register the reference patch there + reset its LSTM state + accumulation.
#
# Run with the SYSTEM python (conda shadows ROS's rclpy):
#   PYTHONNOUSERSITE=1 /usr/bin/python3 \
#     install/libcaer_driver_eventtrack/lib/libcaer_driver_eventtrack/reinit_click_node.py

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Empty


class ReinitClickNode(Node):
    def __init__(self):
        super().__init__('reinit_click')
        self.image_topic = self.declare_parameter(
            'image_topic', '/event_camera/events_rep_image').value
        self.point_topic = self.declare_parameter(
            'point_topic', '/blinktrack_tracker/reinit_point').value
        self.log_topic = self.declare_parameter(
            'log_topic', '/blinktrack_tracker/log_trigger').value

        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        self._frame = None  # latest BGR image (numpy)
        self.sub = self.create_subscription(Image, self.image_topic, self._on_img, qos)
        self.pub = self.create_publisher(PointStamped, self.point_topic, 10)
        self.log_pub = self.create_publisher(Empty, self.log_topic, 10)

        self.win = 'events_rep  [left-click = re-init,  d = log RL I/O 0.5s]'
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)
        self.timer = self.create_timer(1.0 / 30.0, self._on_timer)  # GUI pump

        self.get_logger().info(
            f'reinit_click: showing {self.image_topic}, click -> {self.point_topic}')

    def _on_img(self, msg):
        if msg.encoding not in ('rgb8', 'bgr8'):
            self.get_logger().warn(f'expected rgb8/bgr8, got {msg.encoding}', once=True)
            return
        arr = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, 3)
        # cv2 displays BGR; the vis node publishes rgb8 -> swap channels
        self._frame = arr[:, :, ::-1].copy() if msg.encoding == 'rgb8' else arr.copy()

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = PointStamped()
            pt.header.stamp = self.get_clock().now().to_msg()
            pt.header.frame_id = 'camera'
            pt.point.x = float(x)
            pt.point.y = float(y)
            pt.point.z = 0.0
            self.pub.publish(pt)
            self.get_logger().info(f're-init click at ({x}, {y})')

    def _on_timer(self):
        if self._frame is not None:
            cv2.imshow(self.win, self._frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            self.log_pub.publish(Empty())
            self.get_logger().info('logging trigger sent (d)')


def main(args=None):
    rclpy.init(args=args)
    node = ReinitClickNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
