# -----------------------------------------------------------------------------
# Copyright 2023 Bernd Pfrommer <bernd.pfrommer@gmail.com>
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
#

import launch
from launch.actions import DeclareLaunchArgument as LaunchArg
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration as LaunchConfig
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_setup(context, *args, **kwargs):
    """Create simple node."""
    node = Node(
        package='libcaer_driver_eventtrack',
        executable='driver_node',
        output='screen',
        # prefix=["xterm -e gdb -ex run --args"],
        name=LaunchConfig('camera_name'),
        parameters=[
            {
                'device_type': LaunchConfig('device_type'),
                'device_id': 1,
                'master': LaunchConfig('master'),
                'serial': '',
                'statistics_print_interval': 2.0,
                'camerainfo_url': '',
                'frame_id': '',
                'event_message_time_threshold': 1.0e-3,
                # ---- event-frame (TimeSurface, ~/events_rep) settings ----
                'time_surface_enabled': ParameterValue(
                    LaunchConfig('time_surface_enabled'), value_type=bool
                ),
                'time_surface_window_us': ParameterValue(
                    LaunchConfig('time_surface_window_us'), value_type=int
                ),
                'time_surface_n_bins': ParameterValue(
                    LaunchConfig('time_surface_n_bins'), value_type=int
                ),
                'time_surface_timer_driven': ParameterValue(
                    LaunchConfig('time_surface_timer_driven'), value_type=bool
                ),
                'packet_interval_us': ParameterValue(
                    LaunchConfig('packet_interval_us'), value_type=int
                ),
                'dvs_enabled': True,
                # safe to enable imu
                'imu_accel_enabled': True,
                'imu_gyro_enabled': True,
                # aps affects event quality, disable when not needed.
                # exposes ~/image_raw (DAVIS grayscale) when True; DAVIS only.
                'aps_enabled': ParameterValue(
                    LaunchConfig('aps_enabled'), value_type=bool
                ),
                # other example settings
                # "aps_exposure": 4000,
                # "aps_frame_interval": 40000,
                # "auto_exposure_enabled": False,
                # "auto_exposure_illumination": 127,
                #
                # "subsample_enabled": False,
                # "subsample_horizontal": 3,
                # "bias_sensitivity": 2,  # for dvxplorer
                # "OFFBn_coarse": 4,  # for DAVIS
                # "OFFBn_fine": 0,  # for DAVIS
            },
        ],
        remappings=[('~/reset_timestamps', LaunchConfig('reset_topic'))],
    )
    return [node]


def generate_launch_description():
    """Create simple node by calling opaque function."""
    return launch.LaunchDescription(
        [
            LaunchArg('camera_name', default_value=['event_camera'], description='camera name'),
            LaunchArg(
                'device_type',
                default_value=['davis'],
                description='device type (davis, dvxplorer...)',
            ),
            LaunchArg(
                'master',
                default_value=['True'],
                description='set to true for this camera to be master',
            ),
            LaunchArg(
                'reset_topic',
                default_value=['~/reset_timestamps'],
                description='on the slave, set this to the masters reset topic',
            ),
            # ---- event-frame (TimeSurface, ~/events_rep) arguments ----
            LaunchArg(
                'time_surface_enabled',
                default_value=['True'],
                description='enable/disable the ~/events_rep TimeSurface publisher',
            ),
            LaunchArg(
                'time_surface_window_us',
                default_value=['3000'],
                description='temporal window / frame period in microseconds',
            ),
            LaunchArg(
                'time_surface_n_bins',
                default_value=['5'],
                description='number of temporal bins (channels = 2 * n_bins)',
            ),
            LaunchArg(
                'time_surface_timer_driven',
                default_value=['True'],
                description='publish at a fixed rate (True) vs event-driven (False)',
            ),
            LaunchArg(
                'packet_interval_us',
                default_value=['10000'],
                description='libcaer packet-container flush interval in microseconds',
            ),
            LaunchArg(
                'aps_enabled',
                default_value=['False'],
                description='enable APS grayscale frames on ~/image_raw (DAVIS only)',
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
