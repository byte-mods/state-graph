"""Comprehensive functional tests for the Reinforcement Learning tab.

Tests are split into:
1. Tests that work without SB3 (listings, custom_grid env, error cases)
2. Tests that require SB3 + gymnasium (CartPole training, algorithms, save/load)
"""

import os
import shutil
import time

import pytest
from fastapi.testclient import TestClient
from state_graph.server.app import app, engine

# ---------------------------------------------------------------------------
# Module-level imports for optional deps
# ---------------------------------------------------------------------------

try:
    import gymnasium  # noqa: F401
    import stable_baselines3  # noqa: F401
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

needs_sb3 = pytest.mark.skipif(not HAS_SB3, reason="stable_baselines3 / gymnasium not installed")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

client = TestClient(app)

RL_SAVE_DIR = "./sg_outputs/test_rl_model"


def _reset_rl_engine():
    """Reset the global RL engine to a clean state."""
    import state_graph.server.app as app_mod
    from state_graph.rl.engine import RLEngine

    if app_mod._rl_engine is not None:
        # Stop any ongoing training first
        if app_mod._rl_engine._is_training:
            app_mod._rl_engine.stop_training()
        app_mod._rl_engine = RLEngine()


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset both the main engine and the RL engine between tests."""
    engine.reset()
    _reset_rl_engine()
    yield
    engine.reset()
    _reset_rl_engine()


@pytest.fixture()
def cleanup_saved_model():
    """Remove saved model artifacts after test."""
    yield
    if os.path.exists(RL_SAVE_DIR + ".zip"):
        os.remove(RL_SAVE_DIR + ".zip")
    if os.path.isdir(RL_SAVE_DIR):
        shutil.rmtree(RL_SAVE_DIR)


def wait_for_rl_training(timeout=60):
    """Poll /api/rl/info until training finishes or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        resp = client.get("/api/rl/info")
        if not resp.json().get("is_training", False):
            return resp.json()
        time.sleep(0.5)
    raise TimeoutError("RL training didn't finish within timeout")


# ===========================================================================
# 1. Listing endpoints — no optional deps needed
# ===========================================================================


class TestRLListings:
    """Test environment and algorithm listing endpoints."""

    def test_list_envs(self):
        resp = client.get("/api/rl/envs")
        assert resp.status_code == 200
        data = resp.json()
        assert "environments" in data
        assert "classic" in data["environments"]
        assert "CartPole-v1" in data["environments"]["classic"]

    def test_list_algorithms(self):
        resp = client.get("/api/rl/algorithms")
        assert resp.status_code == 200
        data = resp.json()
        assert "algorithms" in data
        for algo in ["PPO", "A2C", "DQN", "SAC", "TD3", "DDPG"]:
            assert algo in data["algorithms"]

    def test_algorithm_params(self):
        resp = client.get("/api/rl/algorithms")
        ppo = resp.json()["algorithms"]["PPO"]
        assert "params" in ppo
        assert "learning_rate" in ppo["params"]
        assert ppo["type"] == "on-policy"

    def test_algorithm_dqn_type(self):
        resp = client.get("/api/rl/algorithms")
        dqn = resp.json()["algorithms"]["DQN"]
        assert dqn["type"] == "off-policy"
        assert dqn["action_space"] == "discrete"

    def test_algorithm_sac_continuous(self):
        resp = client.get("/api/rl/algorithms")
        sac = resp.json()["algorithms"]["SAC"]
        assert sac["action_space"] == "continuous"

    def test_info_initial(self):
        resp = client.get("/api/rl/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_model"] is False
        assert data["is_training"] is False
        assert data["env_id"] == ""
        assert data["algorithm"] == ""

    def test_history_empty(self):
        resp = client.get("/api/rl/history")
        assert resp.status_code == 200
        assert resp.json()["history"] == []

    def test_env_categories(self):
        resp = client.get("/api/rl/envs")
        envs = resp.json()["environments"]
        assert "classic" in envs
        assert "box2d" in envs
        assert "mujoco" in envs
        assert "pybullet" in envs

    def test_env_classic_entries(self):
        resp = client.get("/api/rl/envs")
        classic = resp.json()["environments"]["classic"]
        for env_name in ["CartPole-v1", "MountainCar-v0", "Acrobot-v1",
                         "LunarLander-v3", "Pendulum-v1", "MountainCarContinuous-v0"]:
            assert env_name in classic

    def test_env_has_metadata(self):
        resp = client.get("/api/rl/envs")
        cartpole = resp.json()["environments"]["classic"]["CartPole-v1"]
        assert "description" in cartpole
        assert "type" in cartpole
        assert cartpole["type"] == "discrete"


# ===========================================================================
# 2. Custom Grid Environment — no SB3 needed
# ===========================================================================


class TestCustomGridEnv:
    """Test the custom grid world environment (no SB3 dependency)."""

    def test_create_custom_grid(self):
        resp = client.post("/api/rl/env", json={
            "env_id": "custom_grid",
            "params": {"size": 5, "n_obstacles": 3, "max_steps": 50},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert data["type"] == "discrete"
        assert data["env_id"] == "custom_grid"

    def test_create_custom_grid_default(self):
        resp = client.post("/api/rl/env", json={"env_id": "custom_grid"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert data["obs_space"] == 4
        assert data["action_space"] == 4

    def test_create_custom_grid_large(self):
        resp = client.post("/api/rl/env", json={
            "env_id": "custom_grid",
            "params": {"size": 20, "n_obstacles": 15, "max_steps": 500},
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"

    def test_info_after_custom_grid(self):
        client.post("/api/rl/env", json={"env_id": "custom_grid"})
        resp = client.get("/api/rl/info")
        data = resp.json()
        assert data["env_id"] == "custom_grid"
        assert data["has_model"] is False


# ===========================================================================
# 3. Error cases — mostly no SB3 needed
# ===========================================================================


class TestRLErrors:
    """Test error handling for invalid operations."""

    def test_episode_without_model(self):
        resp = client.post("/api/rl/episode")
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"
        assert "message" in resp.json()

    def test_save_without_model(self):
        resp = client.post("/api/rl/save", json={"path": "./test_no_model"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_train_without_agent(self):
        resp = client.post("/api/rl/train", json={"total_timesteps": 100})
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_stop_when_not_training(self):
        resp = client.post("/api/rl/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_training"


# ===========================================================================
# 4. SB3 Training Flow — requires stable_baselines3 + gymnasium
# ===========================================================================


@needs_sb3
class TestRLTrainingFlow:
    """End-to-end training flow tests with CartPole + PPO."""

    def test_create_cartpole_env(self):
        resp = client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert data["type"] == "discrete"
        assert data["env_id"] == "CartPole-v1"

    def test_create_ppo_agent(self):
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        resp = client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"n_steps": 64, "batch_size": 32},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert data["total_params"] > 0
        assert data["algorithm"] == "PPO"

    def test_info_after_agent_creation(self):
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"n_steps": 64, "batch_size": 32},
        })
        resp = client.get("/api/rl/info")
        data = resp.json()
        assert data["env_id"] == "CartPole-v1"
        assert data["algorithm"] == "PPO"
        assert data["has_model"] is True
        assert data["is_training"] is False

    def test_full_ppo_cartpole_training(self):
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"n_steps": 64, "batch_size": 32},
        })
        resp = client.post("/api/rl/train", json={
            "total_timesteps": 500,
            "eval_freq": 100,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
        # Wait for completion
        info = wait_for_rl_training(timeout=30)
        assert info["has_model"] is True
        assert info["is_training"] is False

    def test_run_episode_after_training(self):
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"n_steps": 64, "batch_size": 32},
        })
        client.post("/api/rl/train", json={"total_timesteps": 500})
        wait_for_rl_training(timeout=30)

        resp = client.post("/api/rl/episode")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "total_reward" in data
        assert "frames" in data
        assert len(data["frames"]) > 0
        assert "steps" in data
        assert data["steps"] > 0

    def test_episode_frame_structure(self):
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"n_steps": 64, "batch_size": 32},
        })
        client.post("/api/rl/train", json={"total_timesteps": 500})
        wait_for_rl_training(timeout=30)

        resp = client.post("/api/rl/episode")
        frames = resp.json()["frames"]
        assert len(frames) > 0
        frame = frames[0]
        assert "step" in frame
        assert "obs" in frame
        assert "action" in frame
        assert "reward" in frame

    def test_training_history(self):
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"n_steps": 64, "batch_size": 32},
        })
        client.post("/api/rl/train", json={"total_timesteps": 500})
        wait_for_rl_training(timeout=30)

        resp = client.get("/api/rl/history")
        assert resp.status_code == 200
        assert isinstance(resp.json()["history"], list)

    def test_save_and_load_model(self, cleanup_saved_model):
        os.makedirs("./sg_outputs", exist_ok=True)

        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"n_steps": 64, "batch_size": 32},
        })
        client.post("/api/rl/train", json={"total_timesteps": 500})
        wait_for_rl_training(timeout=30)

        # Save
        resp = client.post("/api/rl/save", json={"path": RL_SAVE_DIR})
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"
        assert resp.json()["path"] == RL_SAVE_DIR

        # Load back
        resp = client.post("/api/rl/load", json={
            "path": RL_SAVE_DIR,
            "algorithm": "PPO",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "loaded"

    def test_load_then_run_episode(self, cleanup_saved_model):
        os.makedirs("./sg_outputs", exist_ok=True)

        # Train and save
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"n_steps": 64, "batch_size": 32},
        })
        client.post("/api/rl/train", json={"total_timesteps": 500})
        wait_for_rl_training(timeout=30)
        client.post("/api/rl/save", json={"path": RL_SAVE_DIR})

        # Reset RL engine, then load
        _reset_rl_engine()
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        client.post("/api/rl/load", json={
            "path": RL_SAVE_DIR,
            "algorithm": "PPO",
        })

        # Run episode with loaded model
        resp = client.post("/api/rl/episode")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["total_reward"] > 0

    def test_stop_training(self):
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"n_steps": 128, "batch_size": 64},
        })
        client.post("/api/rl/train", json={"total_timesteps": 100000})
        time.sleep(1)  # Let training start

        resp = client.post("/api/rl/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("stopped", "not_training")

        # Ensure it actually stopped
        time.sleep(1)
        info = client.get("/api/rl/info").json()
        assert info["is_training"] is False

    def test_already_training_guard(self):
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"n_steps": 128, "batch_size": 64},
        })
        client.post("/api/rl/train", json={"total_timesteps": 100000})
        time.sleep(0.5)

        # Try starting training again while already training
        resp = client.post("/api/rl/train", json={"total_timesteps": 100})
        assert resp.json()["status"] == "already_training"

        # Clean up
        client.post("/api/rl/stop")
        wait_for_rl_training(timeout=15)


# ===========================================================================
# 5. Multiple algorithms — requires SB3
# ===========================================================================


@needs_sb3
class TestRLAlgorithms:
    """Test that multiple discrete-action algorithms work with CartPole."""

    @pytest.mark.parametrize("algo", ["PPO", "A2C", "DQN"])
    def test_discrete_algorithms(self, algo):
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        resp = client.post("/api/rl/agent", json={
            "algorithm": algo,
            "params": {"learning_rate": 1e-3},
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"
        assert resp.json()["algorithm"] == algo

        resp = client.post("/api/rl/train", json={"total_timesteps": 300})
        assert resp.json()["status"] == "started"
        info = wait_for_rl_training(timeout=30)
        assert info["is_training"] is False
        assert info["has_model"] is True

    def test_ppo_agent_params_applied(self):
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        resp = client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"learning_rate": 0.01, "n_steps": 32, "batch_size": 16},
        })
        data = resp.json()
        assert data["status"] == "created"
        assert data["params"]["learning_rate"] == 0.01
        assert data["params"]["n_steps"] == 32
        assert data["params"]["batch_size"] == 16

    def test_a2c_agent_creation(self):
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        resp = client.post("/api/rl/agent", json={
            "algorithm": "A2C",
            "params": {},
        })
        assert resp.json()["status"] == "created"
        assert resp.json()["total_params"] > 0

    def test_dqn_agent_creation(self):
        client.post("/api/rl/env", json={"env_id": "CartPole-v1"})
        resp = client.post("/api/rl/agent", json={
            "algorithm": "DQN",
            "params": {"buffer_size": 1000, "batch_size": 32},
        })
        assert resp.json()["status"] == "created"


# ===========================================================================
# 6. Custom grid + SB3 training — requires SB3
# ===========================================================================


@needs_sb3
class TestCustomGridTraining:
    """Test training on the custom grid world using SB3 agents."""

    def test_custom_grid_with_ppo(self):
        client.post("/api/rl/env", json={
            "env_id": "custom_grid",
            "params": {"size": 5, "n_obstacles": 2, "max_steps": 50},
        })
        resp = client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"n_steps": 64, "batch_size": 32},
        })
        assert resp.json()["status"] == "created"

        resp = client.post("/api/rl/train", json={"total_timesteps": 300})
        assert resp.json()["status"] == "started"
        info = wait_for_rl_training(timeout=30)
        assert info["is_training"] is False

    def test_custom_grid_episode(self):
        client.post("/api/rl/env", json={
            "env_id": "custom_grid",
            "params": {"size": 5, "n_obstacles": 2, "max_steps": 50},
        })
        client.post("/api/rl/agent", json={
            "algorithm": "PPO",
            "params": {"n_steps": 64, "batch_size": 32},
        })
        client.post("/api/rl/train", json={"total_timesteps": 300})
        wait_for_rl_training(timeout=30)

        resp = client.post("/api/rl/episode")
        data = resp.json()
        assert data["status"] == "ok"
        assert "total_reward" in data
        assert len(data["frames"]) > 0
        # Custom grid frames have grid-specific fields
        frame = data["frames"][0]
        assert "agent" in frame
        assert "goal" in frame
        assert "size" in frame
        assert "action" in frame
