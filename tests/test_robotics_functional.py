"""Comprehensive functional tests for the Robotics tab.

Covers: component catalog, templates, robot CRUD, component management,
circuit analysis, 3D scene generation, joint control, physics simulation,
and full robot-to-simulation deployment flow.
"""

import time

import pytest
from fastapi.testclient import TestClient

from state_graph.server.app import app, engine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_state():
    """Reset engine, robot manager, and physics server between tests."""
    engine.reset()
    from state_graph.server.app import _robot_mgr, _physics

    _robot_mgr.robots.clear()
    # Reset physics server state
    _physics._running = False
    _physics._bodies = []
    _physics._joints = []
    _physics._state = {}
    _physics._sim_time = 0.0
    yield
    engine.reset()
    _robot_mgr.robots.clear()
    _physics._running = False
    _physics._bodies = []
    _physics._joints = []
    _physics._state = {}
    _physics._sim_time = 0.0


client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Component Catalog
# ---------------------------------------------------------------------------


class TestComponentCatalog:
    def test_list_components(self):
        resp = client.get("/api/robotics/components")
        assert resp.status_code == 200
        data = resp.json()["components"]
        assert "actuator" in data
        assert "sensor" in data
        assert "power" in data
        assert "controller" in data
        assert "structure" in data
        assert "electronics" in data

    def test_component_count(self):
        resp = client.get("/api/robotics/components")
        total = sum(len(v) for v in resp.json()["components"].values())
        assert total == 27

    def test_component_fields(self):
        resp = client.get("/api/robotics/components")
        for category, components in resp.json()["components"].items():
            for comp in components:
                assert "id" in comp
                assert "name" in comp
                assert "specs" in comp
                assert "dimensions" in comp
                assert "color" in comp


# ---------------------------------------------------------------------------
# 2. Templates
# ---------------------------------------------------------------------------


class TestRobotTemplates:
    def test_list_templates(self):
        resp = client.get("/api/robotics/templates")
        assert resp.status_code == 200
        templates = resp.json()["templates"]
        assert len(templates) == 5
        for name in [
            "wheeled_2wd",
            "robot_arm_3dof",
            "quadruped",
            "humanoid",
            "drone_quad",
        ]:
            assert name in templates

    @pytest.mark.parametrize(
        "template",
        ["wheeled_2wd", "robot_arm_3dof", "quadruped", "humanoid", "drone_quad"],
    )
    def test_create_from_each_template(self, template):
        resp = client.post(
            "/api/robotics/robots",
            json={"name": f"Test {template}", "template": template},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert data["robot"]["component_count"] > 0


# ---------------------------------------------------------------------------
# 3. Robot CRUD
# ---------------------------------------------------------------------------


class TestRobotCRUD:
    def test_create_blank_robot(self):
        resp = client.post("/api/robotics/robots", json={"name": "Empty Bot"})
        assert resp.json()["status"] == "created"

    def test_create_with_template(self):
        resp = client.post(
            "/api/robotics/robots",
            json={"name": "My Bot", "template": "wheeled_2wd"},
        )
        assert resp.json()["status"] == "created"
        rid = resp.json()["robot"]["id"]
        # Get it back
        resp = client.get(f"/api/robotics/robots/{rid}")
        assert resp.status_code == 200

    def test_list_robots(self):
        client.post(
            "/api/robotics/robots",
            json={"name": "Bot1", "template": "wheeled_2wd"},
        )
        client.post("/api/robotics/robots", json={"name": "Bot2"})
        resp = client.get("/api/robotics/robots")
        assert len(resp.json()["robots"]) == 2

    def test_delete_robot(self):
        resp = client.post("/api/robotics/robots", json={"name": "Delete Me"})
        rid = resp.json()["robot"]["id"]
        resp = client.delete(f"/api/robotics/robots/{rid}")
        assert resp.json()["status"] == "deleted"

    def test_get_nonexistent_robot(self):
        resp = client.get("/api/robotics/robots/nonexistent")
        assert resp.json().get("status") == "error"


# ---------------------------------------------------------------------------
# 4. Component Management
# ---------------------------------------------------------------------------


class TestComponentManagement:
    def _create_robot(self):
        resp = client.post("/api/robotics/robots", json={"name": "Test Bot"})
        return resp.json()["robot"]["id"]

    def test_add_component(self):
        rid = self._create_robot()
        resp = client.post(
            f"/api/robotics/robots/{rid}/components",
            json={
                "type": "servo_micro",
                "position": [0.05, 0.1, 0],
                "role": "joint_1",
                "duty_cycle": 0.5,
            },
        )
        assert resp.json()["status"] == "added"

    def test_add_unknown_component(self):
        rid = self._create_robot()
        resp = client.post(
            f"/api/robotics/robots/{rid}/components",
            json={"type": "nonexistent_part", "position": [0, 0, 0]},
        )
        assert resp.json()["status"] == "error"

    def test_remove_component(self):
        rid = self._create_robot()
        resp = client.post(
            f"/api/robotics/robots/{rid}/components",
            json={"type": "servo_micro", "position": [0, 0, 0]},
        )
        cid = resp.json()["id"]
        resp = client.delete(f"/api/robotics/robots/{rid}/components/{cid}")
        assert resp.json()["status"] == "removed"

    def test_update_component(self):
        rid = self._create_robot()
        resp = client.post(
            f"/api/robotics/robots/{rid}/components",
            json={"type": "servo_micro", "position": [0, 0, 0]},
        )
        cid = resp.json()["id"]
        resp = client.put(
            f"/api/robotics/robots/{rid}/components/{cid}",
            json={"position": [0.1, 0.2, 0.3], "duty_cycle": 0.8},
        )
        assert resp.json()["status"] == "updated"

    def test_add_multiple_component_types(self):
        rid = self._create_robot()
        components = [
            {"type": "frame_plate", "position": [0, 0, 0], "role": "base"},
            {"type": "servo_standard", "position": [0, 0.05, 0], "role": "hip"},
            {"type": "servo_micro", "position": [0, 0.1, 0], "role": "knee"},
            {"type": "arduino_nano", "position": [0, 0, 0.05], "role": "controller"},
            {"type": "lipo_2s_2200", "position": [0, 0, -0.05], "role": "battery"},
            {"type": "imu_mpu6050", "position": [0, 0, 0], "role": "sensor"},
        ]
        for comp in components:
            resp = client.post(
                f"/api/robotics/robots/{rid}/components", json=comp
            )
            assert resp.json()["status"] == "added"


# ---------------------------------------------------------------------------
# 5. Circuit Analysis
# ---------------------------------------------------------------------------


class TestCircuitAnalysis:
    def test_wheeled_robot_circuit(self):
        resp = client.post(
            "/api/robotics/robots",
            json={"name": "Circuit Test", "template": "wheeled_2wd"},
        )
        rid = resp.json()["robot"]["id"]
        resp = client.get(f"/api/robotics/robots/{rid}/circuit")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_weight_g"] > 0
        assert data["total_current_active_ma"] > 0
        assert data["battery"] is not None
        assert data["battery_life_minutes"] > 0
        assert "voltage_rails" in data
        assert data["current_within_limits"] is True

    def test_quadruped_circuit(self):
        resp = client.post(
            "/api/robotics/robots",
            json={"name": "Quad", "template": "quadruped"},
        )
        rid = resp.json()["robot"]["id"]
        circuit = client.get(f"/api/robotics/robots/{rid}/circuit").json()
        assert circuit["total_weight_g"] > 0
        assert circuit["battery"] is not None

    @pytest.mark.parametrize(
        "template",
        ["wheeled_2wd", "robot_arm_3dof", "quadruped", "humanoid", "drone_quad"],
    )
    def test_all_templates_have_valid_circuits(self, template):
        resp = client.post(
            "/api/robotics/robots",
            json={"name": f"test_{template}", "template": template},
        )
        rid = resp.json()["robot"]["id"]
        circuit = client.get(f"/api/robotics/robots/{rid}/circuit").json()
        assert circuit["total_weight_g"] > 0
        assert len(circuit["components"]) > 0


# ---------------------------------------------------------------------------
# 6. 3D Scene Generation
# ---------------------------------------------------------------------------


class TestSceneGeneration:
    def test_scene_has_objects_and_joints(self):
        resp = client.post(
            "/api/robotics/robots",
            json={"name": "Scene Test", "template": "humanoid"},
        )
        rid = resp.json()["robot"]["id"]
        scene = client.get(f"/api/robotics/robots/{rid}/scene").json()
        assert len(scene["objects"]) > 0
        assert "joints" in scene
        assert "links" in scene
        assert "robot_name" in scene
        assert len(scene["joints"]) > 0  # humanoid has many joints

    def test_scene_object_fields(self):
        resp = client.post(
            "/api/robotics/robots",
            json={"name": "Fields Test", "template": "wheeled_2wd"},
        )
        rid = resp.json()["robot"]["id"]
        scene = client.get(f"/api/robotics/robots/{rid}/scene").json()
        for obj in scene["objects"]:
            assert "id" in obj
            assert "type" in obj
            assert "position" in obj
            assert "dimensions" in obj
            assert "color" in obj
            assert "geometry" in obj

    def test_scene_geometry_types_vary(self):
        resp = client.post(
            "/api/robotics/robots",
            json={"name": "Geo Test", "template": "humanoid"},
        )
        rid = resp.json()["robot"]["id"]
        scene = client.get(f"/api/robotics/robots/{rid}/scene").json()
        geos = set(obj["geometry"] for obj in scene["objects"])
        assert len(geos) > 1  # should have variety

    @pytest.mark.parametrize(
        "template",
        ["wheeled_2wd", "robot_arm_3dof", "quadruped", "humanoid", "drone_quad"],
    )
    def test_all_templates_generate_valid_scenes(self, template):
        resp = client.post(
            "/api/robotics/robots",
            json={"name": f"scene_{template}", "template": template},
        )
        rid = resp.json()["robot"]["id"]
        scene = client.get(f"/api/robotics/robots/{rid}/scene").json()
        assert len(scene["objects"]) > 0


# ---------------------------------------------------------------------------
# 7. Joint Control
# ---------------------------------------------------------------------------


class TestJointControl:
    def _create_arm(self):
        resp = client.post(
            "/api/robotics/robots",
            json={"name": "Arm Test", "template": "robot_arm_3dof"},
        )
        return resp.json()["robot"]["id"]

    def test_get_joints(self):
        rid = self._create_arm()
        resp = client.get(f"/api/robotics/robots/{rid}/joints")
        assert resp.status_code == 200
        joints = resp.json()
        assert len(joints) > 0

    def test_set_single_joint(self):
        rid = self._create_arm()
        joints = client.get(f"/api/robotics/robots/{rid}/joints").json()
        joint_id = list(joints.keys())[0]
        resp = client.post(
            f"/api/robotics/robots/{rid}/joint/{joint_id}",
            json={"angle": 45.0},
        )
        assert resp.json()["status"] == "ok"
        assert resp.json()["angle"] == 45.0

    def test_set_multiple_joints(self):
        rid = self._create_arm()
        joints = client.get(f"/api/robotics/robots/{rid}/joints").json()
        angles = {jid: 30.0 for jid in joints}
        resp = client.post(
            f"/api/robotics/robots/{rid}/joints",
            json={"angles": angles},
        )
        assert resp.json()["status"] == "ok"

    def test_joint_clamping(self):
        rid = self._create_arm()
        joints = client.get(f"/api/robotics/robots/{rid}/joints").json()
        joint_id = list(joints.keys())[0]
        # Set extreme angle -- should be clamped
        resp = client.post(
            f"/api/robotics/robots/{rid}/joint/{joint_id}",
            json={"angle": 999.0},
        )
        assert resp.json()["status"] == "ok"
        assert resp.json()["angle"] <= 180  # clamped

    def test_humanoid_joints(self):
        resp = client.post(
            "/api/robotics/robots",
            json={"name": "Humanoid", "template": "humanoid"},
        )
        rid = resp.json()["robot"]["id"]
        joints = client.get(f"/api/robotics/robots/{rid}/joints").json()
        # Humanoid should have many joints (17 DOF)
        assert len(joints) >= 10


# ---------------------------------------------------------------------------
# 8. Physics Simulation
# ---------------------------------------------------------------------------


class TestPhysicsSimulation:
    def test_load_simple_scene(self):
        resp = client.post(
            "/api/physics/load",
            json={
                "bodies": [
                    {
                        "position": [0, 1, 0],
                        "dimensions": {"x": 0.1, "y": 0.1, "z": 0.1},
                        "mass": 1.0,
                        "name": "box",
                    },
                ],
                "joints": [],
                "use_mujoco": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loaded"
        assert data["bodies"] == 1

    def test_physics_step(self):
        client.post(
            "/api/physics/load",
            json={
                "bodies": [
                    {
                        "position": [0, 1, 0],
                        "dimensions": {"x": 0.1, "y": 0.1, "z": 0.1},
                        "mass": 1.0,
                        "name": "box",
                    },
                ],
                "use_mujoco": False,
            },
        )
        resp = client.post("/api/physics/step")
        assert resp.json()["status"] == "ok"

    def test_physics_info(self):
        client.post(
            "/api/physics/load",
            json={
                "bodies": [
                    {
                        "position": [0, 1, 0],
                        "dimensions": {"x": 0.1, "y": 0.1, "z": 0.1},
                        "mass": 1.0,
                        "name": "box",
                    },
                ],
                "use_mujoco": False,
            },
        )
        resp = client.get("/api/physics/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
        assert "body_count" in data

    def test_apply_force(self):
        client.post(
            "/api/physics/load",
            json={
                "bodies": [
                    {
                        "position": [0, 1, 0],
                        "dimensions": {"x": 0.1, "y": 0.1, "z": 0.1},
                        "mass": 1.0,
                        "name": "box",
                    },
                ],
                "use_mujoco": False,
            },
        )
        resp = client.post(
            "/api/physics/force",
            json={"body_index": 0, "force": [10.0, 0, 0]},
        )
        assert resp.json()["status"] == "ok"

    def test_start_and_stop_simulation(self):
        client.post(
            "/api/physics/load",
            json={
                "bodies": [
                    {
                        "position": [0, 1, 0],
                        "dimensions": {"x": 0.1, "y": 0.1, "z": 0.1},
                        "mass": 1.0,
                        "name": "box",
                    },
                ],
                "use_mujoco": False,
            },
        )
        resp = client.post("/api/physics/start")
        assert resp.json()["status"] == "started"
        time.sleep(0.5)  # let it run a bit
        resp = client.post("/api/physics/stop")
        assert resp.json()["status"] == "stopped"

    def test_multiple_bodies(self):
        client.post(
            "/api/physics/load",
            json={
                "bodies": [
                    {
                        "position": [0, 2, 0],
                        "dimensions": {"x": 0.1, "y": 0.1, "z": 0.1},
                        "mass": 0.5,
                        "name": "box1",
                    },
                    {
                        "position": [0.5, 3, 0],
                        "dimensions": {"x": 0.05, "y": 0.05, "z": 0.05},
                        "mass": 0.2,
                        "name": "box2",
                    },
                    {
                        "position": [-0.5, 1, 0],
                        "dimensions": {"x": 0.2, "y": 0.1, "z": 0.1},
                        "mass": 1.0,
                        "name": "box3",
                    },
                ],
                "use_mujoco": False,
            },
        )
        info = client.get("/api/physics/info").json()
        assert info["body_count"] == 3


# ---------------------------------------------------------------------------
# 9. Full End-to-End: Robot -> Simulation Deployment
# ---------------------------------------------------------------------------


class TestRobotToSimulation:
    """Full workflow: create robot -> configure -> analyze -> deploy to physics -> simulate."""

    def test_full_robot_build_and_simulate(self):
        # Step 1: Create robot from template
        resp = client.post(
            "/api/robotics/robots",
            json={"name": "My Walker", "template": "wheeled_2wd"},
        )
        assert resp.json()["status"] == "created"
        rid = resp.json()["robot"]["id"]

        # Step 2: Add extra sensor
        resp = client.post(
            f"/api/robotics/robots/{rid}/components",
            json={"type": "imu_mpu6050", "position": [0, 0.05, 0], "role": "imu"},
        )
        assert resp.json()["status"] == "added"

        # Step 3: Analyze circuit
        circuit = client.get(f"/api/robotics/robots/{rid}/circuit").json()
        assert circuit["total_weight_g"] > 0
        assert circuit["battery"] is not None
        assert circuit["current_within_limits"] is True

        # Step 4: Get 3D scene
        scene = client.get(f"/api/robotics/robots/{rid}/scene").json()
        assert len(scene["objects"]) > 0

        # Step 5: Deploy to physics engine
        bodies = []
        for obj in scene["objects"]:
            bodies.append(
                {
                    "position": obj["position"],
                    "dimensions": obj["dimensions"],
                    "mass": obj.get("weight_g", 10) / 1000.0,  # grams to kg
                    "name": obj.get("role", obj["type"]),
                }
            )
        resp = client.post(
            "/api/physics/load",
            json={"bodies": bodies, "use_mujoco": False},
        )
        assert resp.json()["status"] == "loaded"

        # Step 6: Run physics steps
        for _ in range(10):
            resp = client.post("/api/physics/step")
            assert resp.json()["status"] == "ok"

        # Step 7: Get physics info
        info = client.get("/api/physics/info").json()
        assert info["body_count"] == len(bodies)

    def test_arm_joint_control_and_simulate(self):
        # Create arm
        resp = client.post(
            "/api/robotics/robots",
            json={"name": "My Arm", "template": "robot_arm_3dof"},
        )
        rid = resp.json()["robot"]["id"]

        # Get joints
        joints = client.get(f"/api/robotics/robots/{rid}/joints").json()
        assert len(joints) > 0

        # Move all joints to 45 degrees
        angles = {jid: 45.0 for jid in joints}
        resp = client.post(
            f"/api/robotics/robots/{rid}/joints",
            json={"angles": angles},
        )
        assert resp.json()["status"] == "ok"

        # Verify joint states
        states = client.get(f"/api/robotics/robots/{rid}/joints").json()
        for jid, state in states.items():
            assert abs(state["angle"] - 45.0) < 0.1

        # Get scene with updated positions
        scene = client.get(f"/api/robotics/robots/{rid}/scene").json()
        assert scene["joint_angles"] is not None

    def test_custom_robot_build_from_scratch(self):
        # Create blank robot
        resp = client.post("/api/robotics/robots", json={"name": "Custom Bot"})
        rid = resp.json()["robot"]["id"]

        # Add chassis
        client.post(
            f"/api/robotics/robots/{rid}/components",
            json={"type": "frame_plate", "position": [0, 0, 0], "role": "chassis"},
        )
        # Add 2 motors + wheels
        client.post(
            f"/api/robotics/robots/{rid}/components",
            json={
                "type": "dc_motor_small",
                "position": [-0.05, 0, 0],
                "role": "left_motor",
            },
        )
        client.post(
            f"/api/robotics/robots/{rid}/components",
            json={
                "type": "dc_motor_small",
                "position": [0.05, 0, 0],
                "role": "right_motor",
            },
        )
        client.post(
            f"/api/robotics/robots/{rid}/components",
            json={
                "type": "wheel_rubber",
                "position": [-0.05, -0.03, 0],
                "role": "left_wheel",
            },
        )
        client.post(
            f"/api/robotics/robots/{rid}/components",
            json={
                "type": "wheel_rubber",
                "position": [0.05, -0.03, 0],
                "role": "right_wheel",
            },
        )
        # Add controller + battery
        client.post(
            f"/api/robotics/robots/{rid}/components",
            json={"type": "esp32", "position": [0, 0.02, 0], "role": "brain"},
        )
        client.post(
            f"/api/robotics/robots/{rid}/components",
            json={
                "type": "lipo_2s_2200",
                "position": [0, 0, -0.03],
                "role": "battery",
            },
        )

        # Verify circuit works
        circuit = client.get(f"/api/robotics/robots/{rid}/circuit").json()
        assert circuit["total_weight_g"] > 0
        assert circuit["battery"] is not None
        assert circuit["battery_life_minutes"] > 0

        # Get scene
        scene = client.get(f"/api/robotics/robots/{rid}/scene").json()
        assert len(scene["objects"]) == 7

    def test_humanoid_full_articulation(self):
        """Test all joints on humanoid can be individually controlled."""
        resp = client.post(
            "/api/robotics/robots",
            json={"name": "Humanoid", "template": "humanoid"},
        )
        rid = resp.json()["robot"]["id"]

        joints = client.get(f"/api/robotics/robots/{rid}/joints").json()

        # Set each joint to a different angle
        for i, jid in enumerate(joints):
            angle = (i + 1) * 10.0  # 10, 20, 30, ...
            resp = client.post(
                f"/api/robotics/robots/{rid}/joint/{jid}",
                json={"angle": angle},
            )
            assert resp.json()["status"] == "ok"

        # Verify all were set
        states = client.get(f"/api/robotics/robots/{rid}/joints").json()
        assert len(states) == len(joints)


# ---------------------------------------------------------------------------
# 10. Issue Detection Tests
# ---------------------------------------------------------------------------


class TestRobotIssueDetection:
    def test_overloaded_battery_warning(self):
        """User adds too many high-power components for the battery."""
        resp = client.post("/api/robotics/robots", json={"name": "Overload"})
        rid = resp.json()["robot"]["id"]
        # Add 10 large DC motors
        for i in range(10):
            client.post(
                f"/api/robotics/robots/{rid}/components",
                json={
                    "type": "dc_motor_large",
                    "position": [i * 0.1, 0, 0],
                    "duty_cycle": 1.0,
                },
            )
        # Add tiny battery
        client.post(
            f"/api/robotics/robots/{rid}/components",
            json={"type": "lipo_1s_500", "position": [0, 0, 0]},
        )
        circuit = client.get(f"/api/robotics/robots/{rid}/circuit").json()
        assert circuit["current_within_limits"] is False

    def test_no_battery_warning(self):
        """Robot without a battery."""
        resp = client.post("/api/robotics/robots", json={"name": "No Battery"})
        rid = resp.json()["robot"]["id"]
        client.post(
            f"/api/robotics/robots/{rid}/components",
            json={"type": "servo_micro", "position": [0, 0, 0]},
        )
        circuit = client.get(f"/api/robotics/robots/{rid}/circuit").json()
        assert circuit["battery"] is None
        assert circuit["battery_life_minutes"] is None
