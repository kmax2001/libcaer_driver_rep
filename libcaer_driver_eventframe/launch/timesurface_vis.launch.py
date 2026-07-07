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
# Standalone launch for the TimeSurface (~/events_rep) -> RGB image visualizer.

import launch
from launch.actions import DeclareLaunchArgument as LaunchArg
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration as LaunchConfig
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_setup(context, *args, **kwargs):
    node = Node(
        package='libcaer_driver_eventframe',
        executable='timesurface_vis_node.py',
        output='screen',
        name='timesurface_vis',
        parameters=[
            {
                'input_topic': LaunchConfig('input_topic'),
                'output_topic': LaunchConfig('output_topic'),
                'rate': ParameterValue(LaunchConfig('rate'), value_type=float),
            },
        ],
    )
    return [node]


def generate_launch_description():
    return launch.LaunchDescription(
        [
            LaunchArg(
                'input_topic',
                default_value=['/event_camera/events_rep'],
                description='TimeSurface topic to subscribe to',
            ),
            LaunchArg(
                'output_topic',
                default_value=['/event_camera/events_rep_image'],
                description='RGB sensor_msgs/Image topic to publish',
            ),
            LaunchArg(
                'rate',
                default_value=['5.0'],
                description='visualization publish rate in fps',
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
