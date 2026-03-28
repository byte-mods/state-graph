"""Tests for robotics simulator, circuit solver, and robot builder."""

import pytest
from state_graph.robotics.simulator import (
    COMPONENT_CATALOG, ROBOT_TEMPLATES, RobotConfig, RobotManager, solve_circuit,
)


class TestComponentCatalog:
    def test_catalog_not_empty(self):
        assert len(COMPONENT_CATALOG) == 27

    def test_all_components_have_required_fields(self):
        for cid, info in COMPONENT_CATALOG.items():
            assert "name" in info, f"{cid} missing name"
            assert "category" in info, f"{cid} missing category"
            assert "specs" in info, f"{cid} missing specs"
            assert "dimensions" in info, f"{cid} missing dimensions"
            assert "color" in info, f"{cid} missing color"

    def test_all_actuators_have_current(self):
        for cid, info in COMPONENT_CATALOG.items():
            if info["category"] == "actuator":
                specs = info["specs"]
                assert "weight_g" in specs, f"{cid} actuator missing weight"

    def test_all_batteries_have_voltage(self):
        for cid, info in COMPONENT_CATALOG.items():
            if info["category"] == "power":
                assert "voltage" in info["specs"]
                assert "capacity_mah" in info["specs"]


class TestCircuitSolver:
    def test_basic_circuit(self):
        components = [
            {"type": "servo_micro", "quantity": 2, "duty_cycle": 0.5},
            {"type": "arduino_nano", "quantity": 1, "duty_cycle": 1.0},
            {"type": "lipo_2s_2200", "quantity": 1},
        ]
        result = solve_circuit(components)
        assert result["total_weight_g"] > 0
        assert result["total_current_active_ma"] > 0
        assert result["battery"] is not None
        assert result["battery_life_minutes"] is not None
        assert result["battery_life_minutes"] > 0

    def test_no_battery(self):
        components = [{"type": "servo_micro", "quantity": 1, "duty_cycle": 0.5}]
        result = solve_circuit(components)
        assert result["battery"] is None
        assert result["battery_life_minutes"] is None

    def test_empty_components(self):
        result = solve_circuit([])
        assert result["total_weight_g"] == 0
        assert result["total_power_w"] == 0

    def test_current_limit_exceeded(self):
        # Many high-power motors on a small battery
        components = [
            {"type": "dc_motor_large", "quantity": 10, "duty_cycle": 1.0},
            {"type": "lipo_1s_500", "quantity": 1},
        ]
        result = solve_circuit(components)
        assert result["current_within_limits"] is False


class TestRobotConfig:
    def test_create_from_template(self):
        r = RobotConfig("Test", "wheeled_2wd")
        assert len(r.components) == 9
        assert r.name == "2WD Wheeled Robot"

    def test_add_component(self):
        r = RobotConfig("Test")
        result = r.add_component("servo_micro", [0, 0.1, 0], "test_joint")
        assert result["status"] == "added"
        assert len(r.components) == 1

    def test_remove_component(self):
        r = RobotConfig("Test")
        res = r.add_component("servo_micro", [0, 0, 0])
        cid = res["id"]
        r.remove_component(cid)
        assert len(r.components) == 0

    def test_analyze_circuit(self):
        r = RobotConfig("Test", "quadruped")
        analysis = r.analyze_circuit()
        assert analysis["total_weight_g"] > 0
        assert "battery" in analysis

    def test_get_3d_scene(self):
        r = RobotConfig("Test", "humanoid")
        scene = r.get_3d_scene()
        assert len(scene["objects"]) > 0
        assert "joints" in scene
        assert "links" in scene
        # Humanoid should have joints
        assert len(scene["joints"]) > 0

    def test_joint_control(self):
        r = RobotConfig("Test", "robot_arm_3dof")
        scene = r.get_3d_scene()
        joint_ids = [j["id"] for j in scene["joints"]]
        assert len(joint_ids) > 0
        # Set a joint angle
        result = r.set_joint_angle(joint_ids[0], 45.0)
        assert result["status"] == "ok"
        assert result["angle"] == 45.0

    def test_joint_limits(self):
        r = RobotConfig("Test", "robot_arm_3dof")
        scene = r.get_3d_scene()
        joint_ids = [j["id"] for j in scene["joints"]]
        # Exceed limits
        result = r.set_joint_angle(joint_ids[0], 999)
        assert result["angle"] <= 180  # Should be clamped

    def test_set_all_joints(self):
        r = RobotConfig("Test", "robot_arm_3dof")
        scene = r.get_3d_scene()
        angles = {j["id"]: 30.0 for j in scene["joints"]}
        result = r.set_all_joints(angles)
        assert result["status"] == "ok"

    def test_get_joint_states(self):
        r = RobotConfig("Test", "robot_arm_3dof")
        r.set_joint_angle(r.components[1]["id"], 45)
        states = r.get_joint_states()
        assert len(states) > 0

    def test_to_dict(self):
        r = RobotConfig("Test", "wheeled_2wd")
        d = r.to_dict()
        assert "total_weight_g" in d
        assert "total_power_w" in d

    def test_geometry_types(self):
        r = RobotConfig("Test", "humanoid")
        scene = r.get_3d_scene()
        geos = set(obj.get("geometry") for obj in scene["objects"])
        # Should have variety
        assert len(geos) > 1


class TestRobotTemplates:
    def test_all_templates_build(self):
        for tid in ROBOT_TEMPLATES:
            r = RobotConfig("Test", tid)
            assert len(r.components) > 0, f"Template {tid} has no components"
            analysis = r.analyze_circuit()
            assert analysis["total_weight_g"] > 0, f"Template {tid} has zero weight"

    def test_template_count(self):
        assert len(ROBOT_TEMPLATES) == 5


class TestRobotManager:
    def test_create_and_list(self):
        mgr = RobotManager()
        r = mgr.create("Test Bot", "wheeled_2wd")
        assert mgr.get(r.id) is not None
        assert len(mgr.list_all()) == 1

    def test_delete(self):
        mgr = RobotManager()
        r = mgr.create("Test")
        mgr.delete(r.id)
        assert mgr.get(r.id) is None
