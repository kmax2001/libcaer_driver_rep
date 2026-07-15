# -----------------------------------------------------------------------------
# Tracker-only container: loads just the BlinkTrackTrackerNode component.
# Use when the driver (or a rosbag) publishes /events_rep from another process.
# Also the smoke test for component loading + the libtorch runtime env.
# -----------------------------------------------------------------------------
import os

import launch
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch.actions import DeclareLaunchArgument as LaunchArg
from launch.actions import OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration as LaunchConfig
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

SITE = os.environ.get(
    "BLINKTRACK_SITE_PACKAGES",
    "/home/kyh/anaconda3/envs/gostop/lib/python3.10/site-packages",
)


def _torch_ld_library_path():
    parts = [
        "/usr/lib/x86_64-linux-gnu",
        f"{SITE}/torch/lib",
        f"{SITE}/tensorrt_libs",
        f"{SITE}/torch_tensorrt/lib",
        "/usr/local/cuda/lib64",
    ]
    return ":".join(parts + [os.environ.get("LD_LIBRARY_PATH", "")])


def launch_setup(context, *args, **kwargs):
    pkg_prefix = get_package_prefix("libcaer_driver_eventtrack")
    pipeline_lib = os.path.join(pkg_prefix, "lib", "libblinktrack_pipeline.so")
    model_dir = os.path.join(
        get_package_share_directory("libcaer_driver_eventtrack"), "models"
    )
    events_topic = LaunchConfig("events_topic").perform(context)
    container = ComposableNodeContainer(
        name="blinktrack_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="libcaer_driver_eventtrack",
                plugin="BlinkTrackTrackerNode",
                name="blinktrack_tracker",
                parameters=[{
                    "pipeline_lib": pipeline_lib,
                    "model_dir": model_dir,
                    "init_x": LaunchConfig("init_x"),
                    "init_y": LaunchConfig("init_y"),
                }],
                remappings=[("/events_rep", events_topic)],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
        output="screen",
    )
    return [container]


def generate_launch_description():
    return launch.LaunchDescription([
        SetEnvironmentVariable("LD_LIBRARY_PATH", _torch_ld_library_path()),
        LaunchArg("events_topic", default_value=["/events_rep"]),
        LaunchArg("init_x", default_value=["126.0"]),
        LaunchArg("init_y", default_value=["37.0"]),
        OpaqueFunction(function=launch_setup),
    ])
