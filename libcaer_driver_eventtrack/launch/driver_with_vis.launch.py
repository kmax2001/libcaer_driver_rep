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
# Convenience launch: bring up the driver (event stream + ~/events_rep
# TimeSurface) AND the TimeSurface -> RGB image visualizer in one shot.
# It simply includes the two existing launch files and forwards their args.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument as LaunchArg
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LaunchConfig


def generate_launch_description():
    pkg = get_package_share_directory('libcaer_driver_eventtrack')
    launch_dir = os.path.join(pkg, 'launch')

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'driver_node.launch.py')),
        launch_arguments={
            'time_surface_enabled': LaunchConfig('time_surface_enabled'),
            'time_surface_window_us': LaunchConfig('time_surface_window_us'),
            'time_surface_n_bins': LaunchConfig('time_surface_n_bins'),
            'time_surface_timer_driven': LaunchConfig('time_surface_timer_driven'),
            'packet_interval_us': LaunchConfig('packet_interval_us'),
        }.items(),
    )

    vis = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'timesurface_vis.launch.py')),
        launch_arguments={
            'input_topic': LaunchConfig('input_topic'),
            'output_topic': LaunchConfig('output_topic'),
            'rate': LaunchConfig('rate'),
        }.items(),
    )

    return LaunchDescription([
        # ---- driver (event-frame) args, defaults match driver_node.launch.py ----
        LaunchArg('time_surface_enabled', default_value=['True']),
        LaunchArg('time_surface_window_us', default_value=['3000']),
        LaunchArg('time_surface_n_bins', default_value=['5']),
        LaunchArg('time_surface_timer_driven', default_value=['True']),
        LaunchArg('packet_interval_us', default_value=['10000']),
        # ---- visualizer args ----
        LaunchArg('input_topic', default_value=['/event_camera/events_rep']),
        LaunchArg('output_topic', default_value=['/event_camera/events_rep_image']),
        LaunchArg('rate', default_value=['5.0']),
        driver,
        vis,
    ])
