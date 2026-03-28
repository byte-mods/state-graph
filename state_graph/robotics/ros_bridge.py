"""ROS2 Bridge — publish/subscribe topics, TF transforms, URDF loading,
sensor messages, action servers, and launch file generation.

Connects StateGraph to the ROS2 ecosystem so trained models can control
real robots via standard ROS interfaces.

Supports: ROS2 Humble/Iron/Jazzy via rclpy.
Also works without ROS2 installed (generates code + launch files for external use).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable


# ── Standard ROS2 Message Types ──

ROS2_MSG_TYPES = {
    "sensor": {
        "sensor_msgs/msg/Image": {"description": "Camera image", "fields": ["header", "height", "width", "encoding", "data"]},
        "sensor_msgs/msg/LaserScan": {"description": "2D LiDAR scan", "fields": ["header", "angle_min", "angle_max", "ranges"]},
        "sensor_msgs/msg/PointCloud2": {"description": "3D point cloud", "fields": ["header", "height", "width", "fields", "data"]},
        "sensor_msgs/msg/Imu": {"description": "IMU data", "fields": ["header", "orientation", "angular_velocity", "linear_acceleration"]},
        "sensor_msgs/msg/JointState": {"description": "Joint positions/velocities/efforts", "fields": ["header", "name", "position", "velocity", "effort"]},
        "sensor_msgs/msg/Range": {"description": "Ultrasonic/IR range", "fields": ["header", "range", "min_range", "max_range"]},
        "sensor_msgs/msg/BatteryState": {"description": "Battery status", "fields": ["header", "voltage", "current", "percentage"]},
        "sensor_msgs/msg/CameraInfo": {"description": "Camera calibration", "fields": ["header", "height", "width", "K", "D"]},
    },
    "geometry": {
        "geometry_msgs/msg/Twist": {"description": "Linear + angular velocity (cmd_vel)", "fields": ["linear", "angular"]},
        "geometry_msgs/msg/Pose": {"description": "Position + orientation", "fields": ["position", "orientation"]},
        "geometry_msgs/msg/PoseStamped": {"description": "Timestamped pose", "fields": ["header", "pose"]},
        "geometry_msgs/msg/Transform": {"description": "TF transform", "fields": ["translation", "rotation"]},
        "geometry_msgs/msg/Wrench": {"description": "Force + torque", "fields": ["force", "torque"]},
    },
    "navigation": {
        "nav_msgs/msg/Odometry": {"description": "Robot odometry", "fields": ["header", "child_frame_id", "pose", "twist"]},
        "nav_msgs/msg/Path": {"description": "Planned path", "fields": ["header", "poses"]},
        "nav_msgs/msg/OccupancyGrid": {"description": "2D map", "fields": ["header", "info", "data"]},
    },
    "control": {
        "trajectory_msgs/msg/JointTrajectory": {"description": "Joint trajectory command", "fields": ["header", "joint_names", "points"]},
        "control_msgs/msg/JointTrajectoryControllerState": {"description": "Controller feedback", "fields": ["header", "joint_names", "actual", "desired", "error"]},
    },
    "standard": {
        "std_msgs/msg/String": {"description": "String message", "fields": ["data"]},
        "std_msgs/msg/Float64": {"description": "Float value", "fields": ["data"]},
        "std_msgs/msg/Bool": {"description": "Boolean", "fields": ["data"]},
        "std_msgs/msg/Int32": {"description": "Integer", "fields": ["data"]},
    },
}

# ── Common ROS2 Packages ──

ROS2_PACKAGES = {
    "navigation": {
        "name": "Nav2 (Navigation2)",
        "description": "Autonomous navigation stack — path planning, obstacle avoidance, SLAM",
        "packages": ["nav2_bringup", "nav2_bt_navigator", "nav2_planner", "nav2_controller"],
        "install": "sudo apt install ros-${ROS_DISTRO}-navigation2 ros-${ROS_DISTRO}-nav2-bringup",
    },
    "slam": {
        "name": "SLAM Toolbox",
        "description": "Simultaneous Localization and Mapping",
        "packages": ["slam_toolbox"],
        "install": "sudo apt install ros-${ROS_DISTRO}-slam-toolbox",
    },
    "moveit": {
        "name": "MoveIt2",
        "description": "Motion planning for robot arms — inverse kinematics, collision avoidance",
        "packages": ["moveit", "moveit_ros_planning", "moveit_ros_move_group"],
        "install": "sudo apt install ros-${ROS_DISTRO}-moveit",
    },
    "gazebo": {
        "name": "Gazebo (gz-sim)",
        "description": "Physics simulation with ROS2 integration",
        "packages": ["ros_gz_sim", "ros_gz_bridge"],
        "install": "sudo apt install ros-${ROS_DISTRO}-ros-gz",
    },
    "micro_ros": {
        "name": "micro-ROS",
        "description": "ROS2 on microcontrollers (ESP32, Arduino, STM32)",
        "packages": ["micro_ros_agent", "micro_ros_setup"],
        "install": "See https://micro.ros.org/docs/tutorials/core/first_application_rtos/",
    },
    "ros2_control": {
        "name": "ros2_control",
        "description": "Hardware abstraction for robot controllers",
        "packages": ["ros2_control", "ros2_controllers", "joint_state_broadcaster"],
        "install": "sudo apt install ros-${ROS_DISTRO}-ros2-control ros-${ROS_DISTRO}-ros2-controllers",
    },
    "perception": {
        "name": "Perception Pipeline",
        "description": "Image processing, object detection, point cloud processing",
        "packages": ["image_transport", "cv_bridge", "pcl_ros", "vision_opencv"],
        "install": "sudo apt install ros-${ROS_DISTRO}-vision-opencv ros-${ROS_DISTRO}-cv-bridge",
    },
    "robot_localization": {
        "name": "robot_localization",
        "description": "Sensor fusion for odometry (EKF/UKF)",
        "packages": ["robot_localization"],
        "install": "sudo apt install ros-${ROS_DISTRO}-robot-localization",
    },
}


class ROS2Bridge:
    """Bridge between StateGraph and ROS2."""

    def __init__(self):
        self._node = None
        self._publishers: dict[str, Any] = {}
        self._subscribers: dict[str, Any] = {}
        self._received_msgs: dict[str, list] = {}
        self._broadcast: Callable | None = None
        self._loop = None
        self._spin_thread: threading.Thread | None = None
        self._running = False

    def set_broadcast(self, fn: Callable, loop) -> None:
        self._broadcast = fn
        self._loop = loop

    def _emit(self, event: str, data: dict) -> None:
        if self._broadcast and self._loop:
            import asyncio
            asyncio.run_coroutine_threadsafe(self._broadcast(event, data), self._loop)

    def check_ros2(self) -> dict:
        """Check if ROS2 is available on this system."""
        try:
            result = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True, timeout=5)
            distro = os.environ.get("ROS_DISTRO", "unknown")
            return {
                "available": result.returncode == 0,
                "distro": distro,
                "topics": result.stdout.strip().split("\n") if result.returncode == 0 else [],
            }
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {"available": False, "distro": None, "message": "ROS2 not found. Install from https://docs.ros.org/"}

    def init_node(self, node_name: str = "stategraph_bridge") -> dict:
        """Initialize a ROS2 node."""
        try:
            import rclpy
            from rclpy.node import Node

            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node(node_name)
            self._running = True
            self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._spin_thread.start()

            return {"status": "ok", "node_name": node_name}
        except ImportError:
            return {"status": "error", "message": "rclpy not installed. Source your ROS2 workspace first."}

    def _spin_loop(self):
        import rclpy
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.1)

    def shutdown(self) -> dict:
        self._running = False
        if self._node:
            self._node.destroy_node()
            self._node = None
        try:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        return {"status": "shutdown"}

    def list_topics(self) -> dict:
        """List active ROS2 topics."""
        try:
            result = subprocess.run(["ros2", "topic", "list", "-t"], capture_output=True, text=True, timeout=5)
            topics = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split()
                    topics.append({"name": parts[0], "type": parts[1] if len(parts) > 1 else ""})
            return {"status": "ok", "topics": topics}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_nodes(self) -> dict:
        try:
            result = subprocess.run(["ros2", "node", "list"], capture_output=True, text=True, timeout=5)
            return {"status": "ok", "nodes": result.stdout.strip().split("\n")}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_services(self) -> dict:
        try:
            result = subprocess.run(["ros2", "service", "list"], capture_output=True, text=True, timeout=5)
            return {"status": "ok", "services": result.stdout.strip().split("\n")}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def publish(self, topic: str, msg_type: str, data: dict) -> dict:
        """Publish a message to a ROS2 topic via CLI (works without rclpy)."""
        try:
            msg_str = json.dumps(data).replace('"', '\\"')
            cmd = f'ros2 topic pub --once {topic} {msg_type} "{msg_str}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            return {"status": "ok" if result.returncode == 0 else "error", "stderr": result.stderr}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def publish_cmd_vel(self, linear_x: float = 0, angular_z: float = 0, topic: str = "/cmd_vel") -> dict:
        """Publish velocity command (most common robot control)."""
        data = {"linear": {"x": linear_x, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": angular_z}}
        return self.publish(topic, "geometry_msgs/msg/Twist", data)

    def publish_joint_state(self, names: list[str], positions: list[float], topic: str = "/joint_commands") -> dict:
        """Publish joint positions."""
        data = {"name": names, "position": positions, "velocity": [], "effort": []}
        return self.publish(topic, "sensor_msgs/msg/JointState", data)

    def echo_topic(self, topic: str, count: int = 5, timeout: int = 10) -> dict:
        """Read messages from a topic."""
        try:
            result = subprocess.run(
                ["ros2", "topic", "echo", topic, "--once"],
                capture_output=True, text=True, timeout=timeout,
            )
            return {"status": "ok", "data": result.stdout}
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_topic_info(self, topic: str) -> dict:
        try:
            result = subprocess.run(["ros2", "topic", "info", topic, "-v"], capture_output=True, text=True, timeout=5)
            return {"status": "ok", "info": result.stdout}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ── URDF Generation ──

    def generate_urdf(self, robot_config: dict) -> str:
        """Generate URDF XML from a StateGraph robot configuration."""
        components = robot_config.get("components", [])
        robot_name = robot_config.get("name", "sg_robot")

        xml = f'<?xml version="1.0"?>\n<robot name="{robot_name}" xmlns:xacro="http://www.ros.org/wiki/xacro">\n\n'

        # Base link
        xml += '  <link name="base_link">\n'
        xml += '    <visual><geometry><box size="0.1 0.05 0.003"/></geometry>'
        xml += '<material name="gray"><color rgba="0.5 0.5 0.5 1"/></material></visual>\n'
        xml += '    <collision><geometry><box size="0.1 0.05 0.003"/></geometry></collision>\n'
        xml += '    <inertial><mass value="0.1"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>\n'
        xml += '  </link>\n\n'

        prev_link = "base_link"
        for i, comp in enumerate(components):
            from state_graph.robotics.simulator import COMPONENT_CATALOG
            info = COMPONENT_CATALOG.get(comp.get("type", ""), {})
            if not info:
                continue

            link_name = comp.get("role", f"link_{i}")
            d = info.get("dimensions", {"x": 0.02, "y": 0.02, "z": 0.02})
            weight = info.get("specs", {}).get("weight_g", 10) / 1000
            pos = comp.get("position", [0, 0, 0])
            color = info.get("color", "#888888")

            # Convert hex color to RGBA
            r = int(color[1:3], 16) / 255
            g = int(color[3:5], 16) / 255
            b = int(color[5:7], 16) / 255

            # Link
            xml += f'  <link name="{link_name}">\n'
            xml += f'    <visual><geometry><box size="{d["x"]} {d["z"]} {d["y"]}"/></geometry>'
            xml += f'<material name="mat_{i}"><color rgba="{r:.2f} {g:.2f} {b:.2f} 1"/></material></visual>\n'
            xml += f'    <collision><geometry><box size="{d["x"]} {d["z"]} {d["y"]}"/></geometry></collision>\n'
            xml += f'    <inertial><mass value="{weight}"/><inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/></inertial>\n'
            xml += f'  </link>\n'

            # Joint
            jt = info.get("joint_type", "fixed")
            ros_jt = {"revolute": "revolute", "continuous": "continuous"}.get(jt, "fixed")
            xml += f'  <joint name="{link_name}_joint" type="{ros_jt}">\n'
            xml += f'    <parent link="{prev_link}"/>\n'
            xml += f'    <child link="{link_name}"/>\n'
            xml += f'    <origin xyz="{pos[0]} {pos[2]} {pos[1]}"/>\n'
            xml += f'    <axis xyz="0 0 1"/>\n'
            if ros_jt == "revolute":
                limit = info.get("specs", {}).get("range_deg", 180) / 2
                rad = limit * math.pi / 180
                effort = info.get("specs", {}).get("torque_kg_cm", 1) * 0.0981
                xml += f'    <limit lower="-{rad:.3f}" upper="{rad:.3f}" effort="{effort:.2f}" velocity="3.14"/>\n'
            xml += f'  </joint>\n\n'

            if jt in ("revolute", "continuous"):
                prev_link = link_name

        xml += '</robot>\n'
        return xml

    # ── Launch File Generation ──

    def generate_launch_file(self, robot_config: dict, packages: list[str] = None) -> str:
        """Generate a ROS2 launch file for the robot."""
        name = robot_config.get("name", "sg_robot")
        code = f'''"""Auto-generated ROS2 launch file for {name}."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    urdf_file = os.path.join(os.path.dirname(__file__), "{name}.urdf")

    with open(urdf_file, "r") as f:
        robot_description = f.read()

    return LaunchDescription([
        # Robot State Publisher
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{{"robot_description": robot_description}}],
        ),

        # Joint State Publisher GUI (for testing)
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
        ),

        # RViz2
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(os.path.dirname(__file__), "config.rviz")],
        ),
'''

        if packages and "navigation" in packages:
            code += '''
        # Nav2 Navigation
        IncludeLaunchDescription(
            os.path.join(
                get_package_share_directory("nav2_bringup"),
                "launch", "navigation_launch.py"
            ),
        ),
'''
        if packages and "slam" in packages:
            code += '''
        # SLAM Toolbox
        Node(
            package="slam_toolbox",
            executable="async_slam_toolbox_node",
            parameters=[{"use_sim_time": True}],
        ),
'''

        code += '    ])\n'
        return code

    # ── ROS2 Package Generation ──

    def generate_package(self, robot_config: dict, output_dir: str = "./sg_ros2_ws/src") -> dict:
        """Generate a complete ROS2 package for the robot."""
        name = robot_config.get("name", "sg_robot").replace(" ", "_").lower()
        pkg_dir = Path(output_dir) / f"{name}_description"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "launch").mkdir(exist_ok=True)
        (pkg_dir / "urdf").mkdir(exist_ok=True)
        (pkg_dir / "config").mkdir(exist_ok=True)
        (pkg_dir / "meshes").mkdir(exist_ok=True)

        # URDF
        urdf = self.generate_urdf(robot_config)
        (pkg_dir / "urdf" / f"{name}.urdf").write_text(urdf)

        # Launch file
        launch = self.generate_launch_file(robot_config)
        (pkg_dir / "launch" / f"{name}_launch.py").write_text(launch)

        # package.xml
        pkg_xml = f'''<?xml version="1.0"?>
<package format="3">
  <name>{name}_description</name>
  <version>1.0.0</version>
  <description>Auto-generated robot description from StateGraph</description>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher_gui</exec_depend>
  <exec_depend>rviz2</exec_depend>
  <exec_depend>xacro</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
'''
        (pkg_dir / "package.xml").write_text(pkg_xml)

        # CMakeLists.txt
        cmake = f'''cmake_minimum_required(VERSION 3.8)
project({name}_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch urdf config meshes
  DESTINATION share/${{PROJECT_NAME}}
)

ament_package()
'''
        (pkg_dir / "CMakeLists.txt").write_text(cmake)

        # Setup instructions
        setup_md = f'''# {name} ROS2 Package

## Build
```bash
cd sg_ros2_ws
colcon build --packages-select {name}_description
source install/setup.bash
```

## Launch
```bash
ros2 launch {name}_description {name}_launch.py
```

## View in RViz
The launch file auto-opens RViz with the robot model.
Use joint_state_publisher_gui to move joints interactively.
'''
        (pkg_dir / "README.md").write_text(setup_md)

        return {
            "status": "generated",
            "package": f"{name}_description",
            "path": str(pkg_dir),
            "files": [str(f.relative_to(pkg_dir)) for f in pkg_dir.rglob("*") if f.is_file()],
        }

    def get_info(self) -> dict:
        ros_check = self.check_ros2()
        return {
            "ros2_available": ros_check.get("available", False),
            "ros_distro": ros_check.get("distro"),
            "node_active": self._node is not None,
            "publishers": list(self._publishers.keys()),
            "subscribers": list(self._subscribers.keys()),
        }
