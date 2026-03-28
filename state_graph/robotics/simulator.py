"""Robot simulation backend — physics state, circuits, component registry.

The 3D rendering and physics happen in the browser (Three.js + Cannon.js).
This module provides the component database, circuit solver, and robot config API.
"""

from __future__ import annotations

import json
import math
import uuid
from typing import Any


# ── Component Catalog ──

COMPONENT_CATALOG = {
    # --- Actuators ---
    "servo_micro": {
        "name": "Micro Servo (SG90)",
        "category": "actuator",
        "subcategory": "servo",
        "specs": {
            "voltage_min": 4.8, "voltage_max": 6.0, "voltage_nominal": 5.0,
            "current_idle_ma": 10, "current_stall_ma": 650,
            "torque_kg_cm": 1.8, "speed_deg_per_sec": 420,
            "weight_g": 9, "range_deg": 180,
        },
        "dimensions": {"x": 0.023, "y": 0.029, "z": 0.012},  # meters
        "joint_type": "revolute",
        "color": "#3498db",
    },
    "servo_standard": {
        "name": "Standard Servo (MG996R)",
        "category": "actuator",
        "subcategory": "servo",
        "specs": {
            "voltage_min": 4.8, "voltage_max": 7.2, "voltage_nominal": 6.0,
            "current_idle_ma": 20, "current_stall_ma": 2500,
            "torque_kg_cm": 11.0, "speed_deg_per_sec": 300,
            "weight_g": 55, "range_deg": 180,
        },
        "dimensions": {"x": 0.040, "y": 0.054, "z": 0.020},
        "joint_type": "revolute",
        "color": "#2980b9",
    },
    "servo_high_torque": {
        "name": "High Torque Servo (DS3225)",
        "category": "actuator",
        "subcategory": "servo",
        "specs": {
            "voltage_min": 4.8, "voltage_max": 6.8, "voltage_nominal": 6.0,
            "current_idle_ma": 30, "current_stall_ma": 3000,
            "torque_kg_cm": 25.0, "speed_deg_per_sec": 250,
            "weight_g": 68, "range_deg": 270,
        },
        "dimensions": {"x": 0.040, "y": 0.054, "z": 0.020},
        "joint_type": "revolute",
        "color": "#1a5276",
    },
    "dc_motor_small": {
        "name": "DC Motor (N20)",
        "category": "actuator",
        "subcategory": "motor",
        "specs": {
            "voltage_nominal": 6.0, "current_no_load_ma": 50,
            "current_stall_ma": 800, "rpm_no_load": 300,
            "torque_stall_kg_cm": 1.5, "weight_g": 12,
        },
        "dimensions": {"x": 0.024, "y": 0.010, "z": 0.012},
        "joint_type": "continuous",
        "color": "#e74c3c",
    },
    "dc_motor_large": {
        "name": "DC Motor (775)",
        "category": "actuator",
        "subcategory": "motor",
        "specs": {
            "voltage_nominal": 12.0, "current_no_load_ma": 300,
            "current_stall_ma": 12000, "rpm_no_load": 12000,
            "torque_stall_kg_cm": 8.0, "weight_g": 320,
        },
        "dimensions": {"x": 0.066, "y": 0.042, "z": 0.042},
        "joint_type": "continuous",
        "color": "#c0392b",
    },
    "stepper_nema17": {
        "name": "Stepper Motor (NEMA 17)",
        "category": "actuator",
        "subcategory": "stepper",
        "specs": {
            "voltage_nominal": 12.0, "current_per_phase_ma": 1700,
            "holding_torque_kg_cm": 4.2, "step_angle_deg": 1.8,
            "weight_g": 280,
        },
        "dimensions": {"x": 0.042, "y": 0.042, "z": 0.040},
        "joint_type": "revolute",
        "color": "#8e44ad",
    },
    "bldc_motor": {
        "name": "Brushless DC Motor (2212)",
        "category": "actuator",
        "subcategory": "bldc",
        "specs": {
            "voltage_nominal": 11.1, "kv": 920,
            "current_max_a": 15, "thrust_g": 800,
            "weight_g": 47,
        },
        "dimensions": {"x": 0.028, "y": 0.028, "z": 0.030},
        "joint_type": "continuous",
        "color": "#9b59b6",
    },

    # --- Sensors ---
    "imu_mpu6050": {
        "name": "IMU (MPU-6050)",
        "category": "sensor",
        "subcategory": "imu",
        "specs": {
            "voltage_nominal": 3.3, "current_ma": 3.9,
            "axes": 6, "gyro_range_dps": 2000, "accel_range_g": 16,
            "weight_g": 2,
        },
        "dimensions": {"x": 0.020, "y": 0.016, "z": 0.003},
        "color": "#27ae60",
    },
    "ultrasonic_hcsr04": {
        "name": "Ultrasonic Distance (HC-SR04)",
        "category": "sensor",
        "subcategory": "distance",
        "specs": {
            "voltage_nominal": 5.0, "current_ma": 15,
            "range_min_cm": 2, "range_max_cm": 400,
            "weight_g": 9,
        },
        "dimensions": {"x": 0.045, "y": 0.020, "z": 0.015},
        "color": "#2ecc71",
    },
    "lidar_rplidar": {
        "name": "LiDAR (RPLiDAR A1)",
        "category": "sensor",
        "subcategory": "lidar",
        "specs": {
            "voltage_nominal": 5.0, "current_ma": 500,
            "range_m": 12, "scan_rate_hz": 5.5, "angular_resolution_deg": 1.0,
            "weight_g": 170,
        },
        "dimensions": {"x": 0.070, "y": 0.070, "z": 0.041},
        "color": "#1abc9c",
    },
    "camera_module": {
        "name": "Camera Module (OV5647)",
        "category": "sensor",
        "subcategory": "camera",
        "specs": {
            "voltage_nominal": 3.3, "current_ma": 250,
            "resolution": "2592x1944", "fps": 30,
            "fov_deg": 62, "weight_g": 3,
        },
        "dimensions": {"x": 0.025, "y": 0.024, "z": 0.009},
        "color": "#16a085",
    },
    "force_sensor": {
        "name": "Force Sensitive Resistor",
        "category": "sensor",
        "subcategory": "force",
        "specs": {
            "voltage_nominal": 3.3, "current_ma": 1,
            "range_n": 100, "weight_g": 1,
        },
        "dimensions": {"x": 0.020, "y": 0.020, "z": 0.002},
        "color": "#f39c12",
    },
    "encoder_rotary": {
        "name": "Rotary Encoder",
        "category": "sensor",
        "subcategory": "encoder",
        "specs": {
            "voltage_nominal": 5.0, "current_ma": 10,
            "ppr": 600, "weight_g": 30,
        },
        "dimensions": {"x": 0.038, "y": 0.035, "z": 0.030},
        "color": "#d35400",
    },

    # --- Batteries ---
    "lipo_1s_500": {
        "name": "LiPo 1S 500mAh",
        "category": "power",
        "subcategory": "battery",
        "specs": {
            "voltage": 3.7, "capacity_mah": 500,
            "max_discharge_c": 25, "weight_g": 15,
        },
        "dimensions": {"x": 0.040, "y": 0.025, "z": 0.008},
        "color": "#f1c40f",
    },
    "lipo_2s_2200": {
        "name": "LiPo 2S 2200mAh",
        "category": "power",
        "subcategory": "battery",
        "specs": {
            "voltage": 7.4, "capacity_mah": 2200,
            "max_discharge_c": 30, "weight_g": 120,
        },
        "dimensions": {"x": 0.105, "y": 0.035, "z": 0.025},
        "color": "#f39c12",
    },
    "lipo_3s_5000": {
        "name": "LiPo 3S 5000mAh",
        "category": "power",
        "subcategory": "battery",
        "specs": {
            "voltage": 11.1, "capacity_mah": 5000,
            "max_discharge_c": 50, "weight_g": 395,
        },
        "dimensions": {"x": 0.155, "y": 0.050, "z": 0.030},
        "color": "#e67e22",
    },
    "18650_cell": {
        "name": "18650 Li-Ion Cell",
        "category": "power",
        "subcategory": "battery",
        "specs": {
            "voltage": 3.7, "capacity_mah": 2600,
            "max_discharge_c": 5, "weight_g": 45,
        },
        "dimensions": {"x": 0.018, "y": 0.018, "z": 0.065},
        "color": "#d4ac0d",
    },

    # --- Controllers ---
    "arduino_nano": {
        "name": "Arduino Nano",
        "category": "controller",
        "subcategory": "mcu",
        "specs": {
            "voltage_nominal": 5.0, "current_ma": 50,
            "cpu_mhz": 16, "ram_kb": 2, "flash_kb": 32,
            "gpio_pins": 22, "pwm_pins": 6, "adc_pins": 8,
            "weight_g": 7,
        },
        "dimensions": {"x": 0.045, "y": 0.018, "z": 0.008},
        "color": "#00979d",
    },
    "raspberry_pi_4": {
        "name": "Raspberry Pi 4",
        "category": "controller",
        "subcategory": "sbc",
        "specs": {
            "voltage_nominal": 5.0, "current_ma": 3000,
            "cpu_ghz": 1.5, "ram_gb": 4, "gpu": "VideoCore VI",
            "gpio_pins": 40, "weight_g": 46,
        },
        "dimensions": {"x": 0.085, "y": 0.056, "z": 0.017},
        "color": "#c51a4a",
    },
    "esp32": {
        "name": "ESP32",
        "category": "controller",
        "subcategory": "mcu",
        "specs": {
            "voltage_nominal": 3.3, "current_ma": 240,
            "cpu_mhz": 240, "ram_kb": 520, "flash_mb": 4,
            "wifi": True, "bluetooth": True,
            "gpio_pins": 34, "weight_g": 5,
        },
        "dimensions": {"x": 0.048, "y": 0.025, "z": 0.008},
        "color": "#333333",
    },
    "jetson_nano": {
        "name": "NVIDIA Jetson Nano",
        "category": "controller",
        "subcategory": "gpu_sbc",
        "specs": {
            "voltage_nominal": 5.0, "current_ma": 4000,
            "cpu_cores": 4, "gpu_cores": 128, "ram_gb": 4,
            "weight_g": 140,
        },
        "dimensions": {"x": 0.100, "y": 0.080, "z": 0.030},
        "color": "#76b900",
    },

    # --- Structural ---
    "frame_plate": {
        "name": "Aluminum Plate (100x50x3mm)",
        "category": "structure",
        "subcategory": "plate",
        "specs": {
            "material": "aluminum_6061",
            "density_kg_m3": 2700,
            "weight_g": 40.5,
        },
        "dimensions": {"x": 0.100, "y": 0.050, "z": 0.003},
        "color": "#95a5a6",
    },
    "frame_tube": {
        "name": "Carbon Fiber Tube (10x200mm)",
        "category": "structure",
        "subcategory": "tube",
        "specs": {
            "material": "carbon_fiber",
            "density_kg_m3": 1600,
            "weight_g": 12,
        },
        "dimensions": {"x": 0.010, "y": 0.010, "z": 0.200},
        "color": "#2c3e50",
    },
    "wheel_rubber": {
        "name": "Rubber Wheel (65mm)",
        "category": "structure",
        "subcategory": "wheel",
        "specs": {
            "diameter_mm": 65, "width_mm": 26,
            "weight_g": 28, "friction_coeff": 0.8,
        },
        "dimensions": {"x": 0.065, "y": 0.026, "z": 0.065},
        "color": "#1a1a1a",
    },

    # --- Electronics ---
    "motor_driver_l298n": {
        "name": "Motor Driver (L298N)",
        "category": "electronics",
        "subcategory": "driver",
        "specs": {
            "voltage_nominal": 5.0, "voltage_motor_max": 46.0,
            "current_per_channel_a": 2.0, "channels": 2,
            "weight_g": 30,
        },
        "dimensions": {"x": 0.043, "y": 0.043, "z": 0.026},
        "color": "#e74c3c",
    },
    "voltage_regulator": {
        "name": "Voltage Regulator (LM7805)",
        "category": "electronics",
        "subcategory": "regulator",
        "specs": {
            "input_voltage_max": 35.0, "output_voltage": 5.0,
            "max_current_a": 1.5, "weight_g": 3,
        },
        "dimensions": {"x": 0.015, "y": 0.010, "z": 0.005},
        "color": "#7f8c8d",
    },
    "bec_5v_3a": {
        "name": "BEC 5V 3A",
        "category": "electronics",
        "subcategory": "regulator",
        "specs": {
            "input_voltage_min": 6.0, "input_voltage_max": 26.0,
            "output_voltage": 5.0, "max_current_a": 3.0,
            "efficiency_pct": 90, "weight_g": 8,
        },
        "dimensions": {"x": 0.030, "y": 0.020, "z": 0.008},
        "color": "#34495e",
    },
}


# ── Robot Templates ──

ROBOT_TEMPLATES = {
    "wheeled_2wd": {
        "name": "2WD Wheeled Robot",
        "description": "Simple differential-drive mobile robot",
        "components": [
            {"type": "frame_plate", "position": [0, 0.03, 0], "role": "chassis"},
            {"type": "dc_motor_small", "position": [-0.04, 0.01, 0], "role": "left_motor"},
            {"type": "dc_motor_small", "position": [0.04, 0.01, 0], "role": "right_motor"},
            {"type": "wheel_rubber", "position": [-0.05, 0.01, 0], "role": "left_wheel"},
            {"type": "wheel_rubber", "position": [0.05, 0.01, 0], "role": "right_wheel"},
            {"type": "ultrasonic_hcsr04", "position": [0, 0.05, 0.04], "role": "front_sensor"},
            {"type": "arduino_nano", "position": [0, 0.04, 0], "role": "brain"},
            {"type": "motor_driver_l298n", "position": [0, 0.03, -0.02], "role": "driver"},
            {"type": "lipo_2s_2200", "position": [0, 0.02, -0.03], "role": "battery"},
        ],
    },
    "robot_arm_3dof": {
        "name": "3-DOF Robot Arm",
        "description": "Articulated robot arm with 3 servo joints",
        "components": [
            {"type": "frame_plate", "position": [0, 0, 0], "role": "base"},
            {"type": "servo_standard", "position": [0, 0.03, 0], "role": "base_joint"},
            {"type": "frame_tube", "position": [0, 0.13, 0], "role": "link1"},
            {"type": "servo_standard", "position": [0, 0.23, 0], "role": "elbow_joint"},
            {"type": "frame_tube", "position": [0, 0.33, 0], "role": "link2"},
            {"type": "servo_micro", "position": [0, 0.43, 0], "role": "wrist_joint"},
            {"type": "arduino_nano", "position": [0.03, 0.02, 0], "role": "brain"},
            {"type": "lipo_2s_2200", "position": [-0.03, 0.01, 0], "role": "battery"},
        ],
    },
    "quadruped": {
        "name": "Quadruped Walker",
        "description": "4-legged walking robot with 12 servos",
        "components": [
            {"type": "frame_plate", "position": [0, 0.06, 0], "role": "body"},
            {"type": "servo_standard", "position": [-0.05, 0.06, 0.04], "role": "fl_hip"},
            {"type": "servo_standard", "position": [-0.05, 0.06, -0.04], "role": "bl_hip"},
            {"type": "servo_standard", "position": [0.05, 0.06, 0.04], "role": "fr_hip"},
            {"type": "servo_standard", "position": [0.05, 0.06, -0.04], "role": "br_hip"},
            {"type": "servo_micro", "position": [-0.05, 0.03, 0.04], "role": "fl_knee"},
            {"type": "servo_micro", "position": [-0.05, 0.03, -0.04], "role": "bl_knee"},
            {"type": "servo_micro", "position": [0.05, 0.03, 0.04], "role": "fr_knee"},
            {"type": "servo_micro", "position": [0.05, 0.03, -0.04], "role": "br_knee"},
            {"type": "servo_micro", "position": [-0.05, 0.0, 0.04], "role": "fl_ankle"},
            {"type": "servo_micro", "position": [-0.05, 0.0, -0.04], "role": "bl_ankle"},
            {"type": "servo_micro", "position": [0.05, 0.0, 0.04], "role": "fr_ankle"},
            {"type": "servo_micro", "position": [0.05, 0.0, -0.04], "role": "br_ankle"},
            {"type": "imu_mpu6050", "position": [0, 0.07, 0], "role": "imu"},
            {"type": "esp32", "position": [0, 0.07, 0.02], "role": "brain"},
            {"type": "lipo_2s_2200", "position": [0, 0.065, 0], "role": "battery"},
        ],
    },
    "humanoid": {
        "name": "Humanoid (17-DOF)",
        "description": "Bipedal humanoid robot",
        "components": [
            {"type": "frame_plate", "position": [0, 0.30, 0], "role": "torso"},
            # Head
            {"type": "servo_micro", "position": [0, 0.35, 0], "role": "neck"},
            {"type": "camera_module", "position": [0, 0.38, 0.02], "role": "eyes"},
            # Arms
            {"type": "servo_standard", "position": [-0.06, 0.30, 0], "role": "l_shoulder"},
            {"type": "servo_standard", "position": [0.06, 0.30, 0], "role": "r_shoulder"},
            {"type": "servo_micro", "position": [-0.06, 0.22, 0], "role": "l_elbow"},
            {"type": "servo_micro", "position": [0.06, 0.22, 0], "role": "r_elbow"},
            # Hips
            {"type": "servo_high_torque", "position": [-0.03, 0.20, 0], "role": "l_hip_yaw"},
            {"type": "servo_high_torque", "position": [0.03, 0.20, 0], "role": "r_hip_yaw"},
            {"type": "servo_high_torque", "position": [-0.03, 0.18, 0], "role": "l_hip_pitch"},
            {"type": "servo_high_torque", "position": [0.03, 0.18, 0], "role": "r_hip_pitch"},
            # Knees
            {"type": "servo_high_torque", "position": [-0.03, 0.10, 0], "role": "l_knee"},
            {"type": "servo_high_torque", "position": [0.03, 0.10, 0], "role": "r_knee"},
            # Ankles
            {"type": "servo_standard", "position": [-0.03, 0.02, 0], "role": "l_ankle"},
            {"type": "servo_standard", "position": [0.03, 0.02, 0], "role": "r_ankle"},
            # Electronics
            {"type": "imu_mpu6050", "position": [0, 0.28, 0], "role": "imu"},
            {"type": "raspberry_pi_4", "position": [0, 0.25, -0.02], "role": "brain"},
            {"type": "lipo_3s_5000", "position": [0, 0.24, 0.02], "role": "battery"},
            {"type": "bec_5v_3a", "position": [0, 0.26, -0.03], "role": "power_reg"},
        ],
    },
    "drone_quad": {
        "name": "Quadcopter Drone",
        "description": "4-motor aerial robot",
        "components": [
            {"type": "frame_plate", "position": [0, 0, 0], "role": "center_plate"},
            {"type": "frame_tube", "position": [-0.08, 0, 0.08], "role": "arm_fl"},
            {"type": "frame_tube", "position": [0.08, 0, 0.08], "role": "arm_fr"},
            {"type": "frame_tube", "position": [-0.08, 0, -0.08], "role": "arm_bl"},
            {"type": "frame_tube", "position": [0.08, 0, -0.08], "role": "arm_br"},
            {"type": "bldc_motor", "position": [-0.10, 0.02, 0.10], "role": "motor_fl"},
            {"type": "bldc_motor", "position": [0.10, 0.02, 0.10], "role": "motor_fr"},
            {"type": "bldc_motor", "position": [-0.10, 0.02, -0.10], "role": "motor_bl"},
            {"type": "bldc_motor", "position": [0.10, 0.02, -0.10], "role": "motor_br"},
            {"type": "imu_mpu6050", "position": [0, 0.01, 0], "role": "imu"},
            {"type": "esp32", "position": [0, -0.01, 0], "role": "flight_controller"},
            {"type": "lipo_3s_5000", "position": [0, -0.02, 0], "role": "battery"},
        ],
    },
}


def solve_circuit(components: list[dict]) -> dict:
    """Compute power budget, current draw, battery life, total weight.

    Each component dict has: {type, quantity, duty_cycle (0-1)}.
    """
    total_weight_g = 0
    total_current_idle_ma = 0
    total_current_active_ma = 0
    voltage_rails: dict[float, list] = {}
    details = []

    for comp in components:
        ctype = comp["type"]
        qty = comp.get("quantity", 1)
        duty = comp.get("duty_cycle", 1.0)

        if ctype not in COMPONENT_CATALOG:
            continue

        info = COMPONENT_CATALOG[ctype]
        specs = info["specs"]
        weight = specs.get("weight_g", 0) * qty
        total_weight_g += weight

        # Current calculation
        if info["category"] == "actuator":
            idle = specs.get("current_idle_ma", specs.get("current_ma", 0))
            active = specs.get("current_stall_ma", specs.get("current_max_a", 0) * 1000)
            avg_current = idle + (active - idle) * duty * 0.3  # 30% of stall as typical load
        elif info["category"] == "power":
            idle = 0
            avg_current = 0  # Batteries are sources, not loads
        else:
            idle = specs.get("current_ma", 0)
            avg_current = idle

        avg_current *= qty
        idle *= qty
        total_current_idle_ma += idle
        total_current_active_ma += avg_current

        v = specs.get("voltage_nominal", specs.get("voltage", 0))
        if v > 0 and info["category"] != "power":
            voltage_rails.setdefault(v, []).append(ctype)

        details.append({
            "type": ctype,
            "name": info["name"],
            "category": info["category"],
            "quantity": qty,
            "weight_g": weight,
            "current_idle_ma": round(idle),
            "current_active_ma": round(avg_current),
            "voltage": v,
            "power_w": round(v * avg_current / 1000, 3),
        })

    # Find battery
    battery_info = None
    for comp in components:
        ctype = comp["type"]
        if ctype in COMPONENT_CATALOG and COMPONENT_CATALOG[ctype]["category"] == "power":
            specs = COMPONENT_CATALOG[ctype]["specs"]
            battery_info = {
                "type": ctype,
                "name": COMPONENT_CATALOG[ctype]["name"],
                "voltage": specs["voltage"],
                "capacity_mah": specs["capacity_mah"],
                "max_current_a": specs["capacity_mah"] * specs["max_discharge_c"] / 1000,
                "weight_g": specs["weight_g"],
            }
            break

    total_power_w = sum(d["power_w"] for d in details)
    battery_life_min = None
    if battery_info and total_current_active_ma > 0:
        battery_life_min = round(battery_info["capacity_mah"] / total_current_active_ma * 60, 1)

    current_ok = True
    if battery_info and total_current_active_ma / 1000 > battery_info["max_current_a"]:
        current_ok = False

    return {
        "total_weight_g": round(total_weight_g, 1),
        "total_current_idle_ma": round(total_current_idle_ma),
        "total_current_active_ma": round(total_current_active_ma),
        "total_power_w": round(total_power_w, 2),
        "voltage_rails": {str(v): names for v, names in voltage_rails.items()},
        "battery": battery_info,
        "battery_life_minutes": battery_life_min,
        "current_within_limits": current_ok,
        "components": details,
    }


class RobotConfig:
    """A robot configuration with components and circuit analysis."""

    def __init__(self, name: str = "Robot", template: str | None = None):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.components: list[dict] = []

        if template and template in ROBOT_TEMPLATES:
            t = ROBOT_TEMPLATES[template]
            self.name = t["name"]
            for comp in t["components"]:
                self.components.append({
                    "id": str(uuid.uuid4())[:6],
                    "type": comp["type"],
                    "position": comp["position"],
                    "role": comp["role"],
                    "quantity": 1,
                    "duty_cycle": 0.5,
                })

    def add_component(self, comp_type: str, position: list[float], role: str = "", duty_cycle: float = 0.5) -> dict:
        if comp_type not in COMPONENT_CATALOG:
            return {"status": "error", "message": f"Unknown component: {comp_type}"}
        cid = str(uuid.uuid4())[:6]
        self.components.append({
            "id": cid,
            "type": comp_type,
            "position": position,
            "role": role,
            "quantity": 1,
            "duty_cycle": duty_cycle,
        })
        return {"status": "added", "id": cid}

    def remove_component(self, comp_id: str) -> dict:
        self.components = [c for c in self.components if c["id"] != comp_id]
        return {"status": "removed"}

    def update_component(self, comp_id: str, updates: dict) -> dict:
        for c in self.components:
            if c["id"] == comp_id:
                c.update(updates)
                return {"status": "updated"}
        return {"status": "error", "message": "Not found"}

    def analyze_circuit(self) -> dict:
        return solve_circuit(self.components)

    def get_3d_scene(self) -> dict:
        """Generate full 3D scene with articulated joints and link hierarchy."""
        objects = []
        joints = []
        joint_angles = {}

        for comp in self.components:
            info = COMPONENT_CATALOG.get(comp["type"])
            if not info:
                continue

            obj = {
                "id": comp["id"],
                "type": comp["type"],
                "name": info["name"],
                "category": info["category"],
                "subcategory": info.get("subcategory", ""),
                "role": comp.get("role", ""),
                "position": comp["position"],
                "dimensions": info["dimensions"],
                "color": info["color"],
                "weight_g": info["specs"].get("weight_g", 0),
                "geometry": self._get_geometry(info),
            }

            jt = info.get("joint_type")
            if jt:
                obj["joint_type"] = jt
                specs = info["specs"]
                obj["joint_limits"] = {
                    "min_deg": -specs.get("range_deg", 180) / 2,
                    "max_deg": specs.get("range_deg", 180) / 2,
                }
                obj["torque_kg_cm"] = specs.get("torque_kg_cm", specs.get("holding_torque_kg_cm", 0))
                obj["speed_deg_per_sec"] = specs.get("speed_deg_per_sec", 100)
                joint_angles[comp["id"]] = comp.get("joint_angle", 0)

                joints.append({
                    "id": comp["id"],
                    "role": comp.get("role", ""),
                    "type": jt,
                    "position": comp["position"],
                    "angle": comp.get("joint_angle", 0),
                    "limits": obj["joint_limits"],
                    "axis": comp.get("joint_axis", [0, 0, 1]),
                })

            objects.append(obj)

        # Build link hierarchy from roles
        links = self._build_link_chain()

        return {
            "objects": objects,
            "joints": joints,
            "joint_angles": joint_angles,
            "links": links,
            "robot_name": self.name,
        }

    def _get_geometry(self, info: dict) -> str:
        """Determine best geometry type for rendering."""
        subcat = info.get("subcategory", "")
        cat = info["category"]
        if subcat in ("wheel",):
            return "cylinder"
        if subcat in ("tube",):
            return "cylinder"
        if subcat in ("servo", "motor", "stepper", "bldc"):
            return "detailed_box"  # Rendered with bevels and shaft
        if subcat in ("battery",):
            return "rounded_box"
        if cat == "sensor":
            return "sphere" if subcat in ("lidar",) else "small_box"
        return "box"

    def _build_link_chain(self) -> list[dict]:
        """Build parent-child link chain from component roles."""
        chain = []
        roles = [(c["id"], c.get("role", ""), c["position"]) for c in self.components]

        # Simple heuristic: connect joints to nearest structural element
        joints = [r for r in roles if any(kw in r[1] for kw in ["hip", "knee", "ankle", "shoulder", "elbow", "wrist", "joint", "neck"])]
        structures = [r for r in roles if r not in joints]

        for j_id, j_role, j_pos in joints:
            # Find closest structure by Y (vertical)
            parent = None
            min_dist = float("inf")
            for s_id, s_role, s_pos in structures:
                dist = sum((a - b) ** 2 for a, b in zip(j_pos, s_pos)) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    parent = s_id
            chain.append({"child": j_id, "parent": parent, "role": j_role})

        return chain

    def set_joint_angle(self, comp_id: str, angle: float) -> dict:
        """Set a joint angle (degrees)."""
        for c in self.components:
            if c["id"] == comp_id:
                info = COMPONENT_CATALOG.get(c["type"], {})
                jt = info.get("joint_type")
                if not jt:
                    return {"status": "error", "message": "Not a joint"}
                specs = info.get("specs", {})
                max_range = specs.get("range_deg", 360)
                angle = max(-max_range / 2, min(max_range / 2, angle))
                c["joint_angle"] = angle
                return {"status": "ok", "angle": angle, "comp_id": comp_id}
        return {"status": "error", "message": "Component not found"}

    def set_all_joints(self, angles: dict[str, float]) -> dict:
        """Set multiple joint angles at once. {comp_id: angle_deg}"""
        results = {}
        for cid, angle in angles.items():
            r = self.set_joint_angle(cid, angle)
            results[cid] = r.get("angle", angle)
        return {"status": "ok", "angles": results}

    def get_joint_states(self) -> dict:
        """Get current angles of all joints."""
        states = {}
        for c in self.components:
            info = COMPONENT_CATALOG.get(c["type"], {})
            if info.get("joint_type"):
                states[c["id"]] = {
                    "role": c.get("role", ""),
                    "angle": c.get("joint_angle", 0),
                    "type": info["joint_type"],
                }
        return states

    def to_dict(self) -> dict:
        analysis = self.analyze_circuit()
        return {
            "id": self.id,
            "name": self.name,
            "component_count": len(self.components),
            "components": self.components,
            "total_weight_g": analysis["total_weight_g"],
            "total_power_w": analysis["total_power_w"],
            "battery_life_min": analysis.get("battery_life_minutes"),
        }


class RobotManager:
    """Manages robot configurations."""

    def __init__(self):
        self.robots: dict[str, RobotConfig] = {}

    def create(self, name: str = "Robot", template: str | None = None) -> RobotConfig:
        r = RobotConfig(name, template)
        self.robots[r.id] = r
        return r

    def get(self, rid: str) -> RobotConfig | None:
        return self.robots.get(rid)

    def list_all(self) -> list[dict]:
        return [r.to_dict() for r in self.robots.values()]

    def delete(self, rid: str) -> dict:
        if rid in self.robots:
            del self.robots[rid]
            return {"status": "deleted"}
        return {"status": "error"}
