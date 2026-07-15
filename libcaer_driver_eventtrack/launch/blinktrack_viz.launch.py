# -----------------------------------------------------------------------------
# Visualization + click-to-reinit UI for the BlinkTrack tracker.
#   - timesurface_vis: /events_rep -> RGB image (with /track keypoint overlay)
#   - reinit_click:    OpenCV window; left-click publishes ~/reinit_point so the
#                      tracker re-registers the reference + resets LSTM/stack.
# Run alongside blinktrack_tracker.launch.py (driver + tracker).
#
# Both are rclpy nodes forced onto the SYSTEM python (conda shadows ROS rclpy)
# with PYTHONNOUSERSITE=1 (a stale ~/.local numpy would otherwise shadow it).
# -----------------------------------------------------------------------------
import launch
from launch.actions import DeclareLaunchArgument as LaunchArg
from launch.actions import SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration as LaunchConfig
from launch_ros.actions import Node


def generate_launch_description():
    events_image = LaunchConfig("events_image_topic")
    track = LaunchConfig("track_topic")
    reinit = LaunchConfig("reinit_topic")

    vis = Node(
        package="libcaer_driver_eventtrack",
        executable="timesurface_vis_node.py",
        name="timesurface_vis",
        prefix="/usr/bin/python3",
        output="screen",
        parameters=[{
            "input_topic": LaunchConfig("events_rep_topic"),
            "output_topic": events_image,
            "track_topic": track,
            "rate": LaunchConfig("rate"),
        }],
    )
    click = Node(
        package="libcaer_driver_eventtrack",
        executable="reinit_click_node.py",
        name="reinit_click",
        prefix="/usr/bin/python3",
        output="screen",
        parameters=[{
            "image_topic": events_image,
            "point_topic": reinit,
        }],
    )
    return launch.LaunchDescription([
        SetEnvironmentVariable("PYTHONNOUSERSITE", "1"),
        LaunchArg("events_rep_topic", default_value=["/event_camera/events_rep"]),
        LaunchArg("events_image_topic", default_value=["/event_camera/events_rep_image"]),
        LaunchArg("track_topic", default_value=["/blinktrack_tracker/track"]),
        LaunchArg("reinit_topic", default_value=["/blinktrack_tracker/reinit_point"]),
        LaunchArg("rate", default_value=["30.0"], description="viz publish fps"),
        vis,
        click,
    ])
