# -----------------------------------------------------------------------------
# Compose the libcaer driver + the C++ BlinkTrack tracker in ONE
# ComposableNodeContainer so /events_rep is handed to the tracker intra-process
# (zero-copy). The tracker component dlopens the ABI=0 libtorch pipeline with
# RTLD_DEEPBIND; the container process therefore needs libtorch / torch_tensorrt
# on LD_LIBRARY_PATH, injected here via SetEnvironmentVariable.
#
# NOTE: the shipped models carry a reference patch baked from the EC eval
# sequence. Tracking a live camera still needs the reference-patch-from-grayscale
# init (see follow-up); this launch wires up the full data path.
# -----------------------------------------------------------------------------
import os

import launch
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch.actions import DeclareLaunchArgument as LaunchArg
from launch.actions import OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration as LaunchConfig
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

# libtorch / torch_tensorrt / cuda live in the gostop conda env (override with
# the BLINKTRACK_SITE_PACKAGES env var if your install differs).
SITE = os.environ.get(
    "BLINKTRACK_SITE_PACKAGES",
    "/home/kyh/anaconda3/envs/gostop/lib/python3.10/site-packages",
)


def _torch_ld_library_path():
    parts = [
        "/usr/lib/x86_64-linux-gnu",          # system libstdc++ (GLIBCXX_3.4.30) before conda's
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
    camera_name = LaunchConfig("camera_name").perform(context)
    window_us = int(LaunchConfig("time_surface_window_us").perform(context))
    packet_us = int(LaunchConfig("packet_interval_us").perform(context))
    n_bins = int(LaunchConfig("time_surface_n_bins").perform(context))
    use_rl = LaunchConfig("use_rl").perform(context).lower() in ("true", "1", "yes")
    max_accum = int(LaunchConfig("max_accum").perform(context))

    container = ComposableNodeContainer(
        name="blinktrack_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="libcaer_driver_eventtrack",
                plugin="libcaer_driver::Driver",
                name=camera_name,
                parameters=[{
                    "device_type": LaunchConfig("device_type"),
                    "device_id": 1,
                    "encoding": "libcaer_cmp",
                    "event_message_time_threshold": 1.0e-3,
                    "aps_enabled": True,   # APS grayscale for the reference-patch init
                    "dvs_enabled": True,
                    "time_surface_window_us": window_us,   # event-time span per frame
                    "time_surface_n_bins": n_bins,         # channels = 2*n_bins (=10)
                    "packet_interval_us": packet_us,       # < window: multiple packets/frame
                }],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package="libcaer_driver_eventtrack",
                plugin="BlinkTrackTrackerNode",
                name="blinktrack_tracker",
                parameters=[{
                    "pipeline_lib": pipeline_lib,
                    "model_dir": model_dir,
                    "init_x": LaunchConfig("init_x"),
                    "init_y": LaunchConfig("init_y"),
                    "use_camera_reference": True,
                    "use_rl": use_rl,
                    "max_accum": max_accum,
                }],
                # driver publishes ~/events_rep, ~/image_raw under /<camera_name>;
                # tracker subscribes the absolute names -> remap both.
                remappings=[
                    ("/events_rep", f"/{camera_name}/events_rep"),
                    ("/image_raw", f"/{camera_name}/image_raw"),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
        output="screen",
    )
    return [container]


def generate_launch_description():
    return launch.LaunchDescription([
        SetEnvironmentVariable("LD_LIBRARY_PATH", _torch_ld_library_path()),
        LaunchArg("camera_name", default_value=["event_camera"]),
        LaunchArg("device_type", default_value=["davis"]),
        LaunchArg("init_x", default_value=["-1.0"], description="init keypoint x (px); <0 = image center"),
        LaunchArg("init_y", default_value=["-1.0"], description="init keypoint y (px); <0 = image center"),
        LaunchArg("time_surface_window_us", default_value=["5000"],
                  description="event-time span per TimeSurface frame (µs); ~training dt"),
        LaunchArg("packet_interval_us", default_value=["2500"],
                  description="libcaer packet read interval (µs); keep < window"),
        LaunchArg("time_surface_n_bins", default_value=["5"],
                  description="temporal bins; channels = 2*n_bins (model needs 10)"),
        LaunchArg("use_rl", default_value=["true"],
                  description="true: RL gates fire/accumulate; false: run tracker every frame"),
        LaunchArg("max_accum", default_value=["64"],
                  description="ring-buffer accumulation cap (drop oldest beyond N)"),
        OpaqueFunction(function=launch_setup),
    ])
