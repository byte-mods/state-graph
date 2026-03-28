"""RL training engine — Gymnasium + Stable-Baselines3 + custom envs.

Supports: PPO, A2C, DQN, SAC, TD3, DDPG, HER.
Environments: Gymnasium classic, Atari, MuJoCo, PyBullet, custom grid/continuous.
Broadcasts real-time metrics (reward, loss, episode length) via WebSocket.
"""

from __future__ import annotations

import json
import math
import threading
import time
import traceback
import uuid
from typing import Any, Callable


# ── Built-in Environment Registry ──

BUILTIN_ENVS = {
    "classic": {
        "CartPole-v1": {"description": "Balance a pole on a cart", "obs": 4, "actions": 2, "type": "discrete"},
        "MountainCar-v0": {"description": "Drive a car up a mountain", "obs": 2, "actions": 3, "type": "discrete"},
        "Acrobot-v1": {"description": "Swing up a two-link robot", "obs": 6, "actions": 3, "type": "discrete"},
        "LunarLander-v3": {"description": "Land a spacecraft", "obs": 8, "actions": 4, "type": "discrete"},
        "Pendulum-v1": {"description": "Swing up a pendulum", "obs": 3, "actions": "continuous", "type": "continuous"},
        "MountainCarContinuous-v0": {"description": "Continuous mountain car", "obs": 2, "actions": "continuous", "type": "continuous"},
    },
    "box2d": {
        "BipedalWalker-v3": {"description": "2D walking robot", "obs": 24, "actions": "4 continuous", "type": "continuous"},
        "BipedalWalkerHardcore-v3": {"description": "Harder walking with obstacles", "obs": 24, "actions": "4 continuous", "type": "continuous"},
        "CarRacing-v3": {"description": "Top-down car racing", "obs": "96x96x3", "actions": "3 continuous", "type": "continuous"},
    },
    "mujoco": {
        "HalfCheetah-v5": {"description": "2D cheetah robot running", "obs": 17, "actions": "6 continuous", "type": "continuous"},
        "Hopper-v5": {"description": "Single-leg hopping robot", "obs": 11, "actions": "3 continuous", "type": "continuous"},
        "Walker2d-v5": {"description": "2D bipedal walker", "obs": 17, "actions": "6 continuous", "type": "continuous"},
        "Ant-v5": {"description": "4-legged ant robot", "obs": 27, "actions": "8 continuous", "type": "continuous"},
        "Humanoid-v5": {"description": "3D humanoid robot walking", "obs": 376, "actions": "17 continuous", "type": "continuous"},
        "HumanoidStandup-v5": {"description": "Humanoid standing up", "obs": 376, "actions": "17 continuous", "type": "continuous"},
        "Swimmer-v5": {"description": "3-link swimming robot", "obs": 8, "actions": "2 continuous", "type": "continuous"},
        "Reacher-v5": {"description": "2D robot arm reaching target", "obs": 11, "actions": "2 continuous", "type": "continuous"},
        "InvertedPendulum-v5": {"description": "Balance inverted pendulum", "obs": 4, "actions": "1 continuous", "type": "continuous"},
        "InvertedDoublePendulum-v5": {"description": "Double inverted pendulum", "obs": 11, "actions": "1 continuous", "type": "continuous"},
    },
    "pybullet": {
        "HumanoidBulletEnv-v0": {"description": "PyBullet humanoid", "obs": 44, "actions": "17 continuous", "type": "continuous"},
        "AntBulletEnv-v0": {"description": "PyBullet ant", "obs": 28, "actions": "8 continuous", "type": "continuous"},
        "HopperBulletEnv-v0": {"description": "PyBullet hopper", "obs": 15, "actions": "3 continuous", "type": "continuous"},
        "Walker2DBulletEnv-v0": {"description": "PyBullet walker", "obs": 22, "actions": "6 continuous", "type": "continuous"},
        "HalfCheetahBulletEnv-v0": {"description": "PyBullet half-cheetah", "obs": 26, "actions": "6 continuous", "type": "continuous"},
        "ReacherBulletEnv-v0": {"description": "PyBullet reacher arm", "obs": 9, "actions": "2 continuous", "type": "continuous"},
        "KukaBulletEnv-v0": {"description": "Kuka robot arm grasping", "obs": "varies", "actions": "varies", "type": "continuous"},
    },
}

# ── Algorithms ──

ALGORITHMS = {
    "PPO": {
        "name": "Proximal Policy Optimization",
        "type": "on-policy",
        "action_space": "both",
        "description": "Most popular general-purpose RL algorithm",
        "params": {
            "learning_rate": {"default": 3e-4, "type": "float"},
            "n_steps": {"default": 2048, "type": "int"},
            "batch_size": {"default": 64, "type": "int"},
            "n_epochs": {"default": 10, "type": "int"},
            "gamma": {"default": 0.99, "type": "float"},
            "gae_lambda": {"default": 0.95, "type": "float"},
            "clip_range": {"default": 0.2, "type": "float"},
            "ent_coef": {"default": 0.0, "type": "float"},
            "vf_coef": {"default": 0.5, "type": "float"},
            "max_grad_norm": {"default": 0.5, "type": "float"},
        },
    },
    "A2C": {
        "name": "Advantage Actor-Critic",
        "type": "on-policy",
        "action_space": "both",
        "description": "Synchronous advantage actor-critic",
        "params": {
            "learning_rate": {"default": 7e-4, "type": "float"},
            "n_steps": {"default": 5, "type": "int"},
            "gamma": {"default": 0.99, "type": "float"},
            "gae_lambda": {"default": 1.0, "type": "float"},
            "ent_coef": {"default": 0.0, "type": "float"},
            "vf_coef": {"default": 0.5, "type": "float"},
            "max_grad_norm": {"default": 0.5, "type": "float"},
        },
    },
    "DQN": {
        "name": "Deep Q-Network",
        "type": "off-policy",
        "action_space": "discrete",
        "description": "Q-learning with neural network (discrete actions only)",
        "params": {
            "learning_rate": {"default": 1e-4, "type": "float"},
            "buffer_size": {"default": 1000000, "type": "int"},
            "batch_size": {"default": 32, "type": "int"},
            "gamma": {"default": 0.99, "type": "float"},
            "exploration_fraction": {"default": 0.1, "type": "float"},
            "exploration_final_eps": {"default": 0.05, "type": "float"},
            "target_update_interval": {"default": 10000, "type": "int"},
            "train_freq": {"default": 4, "type": "int"},
        },
    },
    "SAC": {
        "name": "Soft Actor-Critic",
        "type": "off-policy",
        "action_space": "continuous",
        "description": "Maximum entropy RL (continuous actions)",
        "params": {
            "learning_rate": {"default": 3e-4, "type": "float"},
            "buffer_size": {"default": 1000000, "type": "int"},
            "batch_size": {"default": 256, "type": "int"},
            "gamma": {"default": 0.99, "type": "float"},
            "tau": {"default": 0.005, "type": "float"},
            "ent_coef": {"default": "auto", "type": "str"},
            "train_freq": {"default": 1, "type": "int"},
        },
    },
    "TD3": {
        "name": "Twin Delayed DDPG",
        "type": "off-policy",
        "action_space": "continuous",
        "description": "Improved DDPG with twin Q-networks",
        "params": {
            "learning_rate": {"default": 1e-3, "type": "float"},
            "buffer_size": {"default": 1000000, "type": "int"},
            "batch_size": {"default": 256, "type": "int"},
            "gamma": {"default": 0.99, "type": "float"},
            "tau": {"default": 0.005, "type": "float"},
            "policy_delay": {"default": 2, "type": "int"},
        },
    },
    "DDPG": {
        "name": "Deep Deterministic Policy Gradient",
        "type": "off-policy",
        "action_space": "continuous",
        "description": "Continuous action space with deterministic policy",
        "params": {
            "learning_rate": {"default": 1e-3, "type": "float"},
            "buffer_size": {"default": 1000000, "type": "int"},
            "batch_size": {"default": 256, "type": "int"},
            "gamma": {"default": 0.99, "type": "float"},
            "tau": {"default": 0.005, "type": "float"},
        },
    },
}


class CustomGridEnv:
    """Simple grid world for quick RL experimentation without Gymnasium dependency."""

    def __init__(self, size: int = 8, n_obstacles: int = 5, max_steps: int = 200):
        self.size = size
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        self.action_space_n = 4  # up, down, left, right
        self.obs_size = 4  # agent_x, agent_y, goal_x, goal_y
        self.reset()

    def reset(self):
        import random
        self.agent = [0, 0]
        self.goal = [self.size - 1, self.size - 1]
        self.obstacles = set()
        while len(self.obstacles) < self.n_obstacles:
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if pos != tuple(self.agent) and pos != tuple(self.goal):
                self.obstacles.add(pos)
        self.steps = 0
        return self._obs()

    def step(self, action):
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x = max(0, min(self.size - 1, self.agent[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent[1] + dy))

        if (new_x, new_y) not in self.obstacles:
            self.agent = [new_x, new_y]

        self.steps += 1
        done = self.agent == self.goal or self.steps >= self.max_steps
        reward = 1.0 if self.agent == self.goal else -0.01
        return self._obs(), reward, done, {}

    def _obs(self):
        return [self.agent[0] / self.size, self.agent[1] / self.size,
                self.goal[0] / self.size, self.goal[1] / self.size]

    def get_state_for_render(self):
        return {
            "agent": self.agent,
            "goal": self.goal,
            "obstacles": list(self.obstacles),
            "size": self.size,
        }


class RLEngine:
    """RL training engine with real-time metric broadcasting."""

    def __init__(self):
        self.model = None
        self.env = None
        self.env_id: str = ""
        self.algorithm: str = ""
        self._is_training = False
        self._stop_event = threading.Event()
        self._train_thread: threading.Thread | None = None
        self._broadcast: Callable | None = None
        self._loop = None
        self._history: list[dict] = []
        self._episode_frames: list[dict] = []  # For replay
        self._custom_env: CustomGridEnv | None = None

    def set_broadcast(self, fn: Callable, loop) -> None:
        self._broadcast = fn
        self._loop = loop

    def _emit(self, event: str, data: dict) -> None:
        if self._broadcast and self._loop:
            import asyncio
            asyncio.run_coroutine_threadsafe(self._broadcast(event, data), self._loop)

    def create_env(self, env_id: str, env_params: dict | None = None) -> dict:
        """Create a Gymnasium environment."""
        self.env_id = env_id

        if env_id == "custom_grid":
            params = env_params or {}
            self._custom_env = CustomGridEnv(
                size=params.get("size", 8),
                n_obstacles=params.get("n_obstacles", 5),
                max_steps=params.get("max_steps", 200),
            )
            return {
                "status": "created",
                "env_id": "custom_grid",
                "obs_space": 4,
                "action_space": 4,
                "type": "discrete",
            }

        import gymnasium as gym
        self.env = gym.make(env_id)
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        return {
            "status": "created",
            "env_id": env_id,
            "obs_space": str(obs_space),
            "action_space": str(act_space),
            "type": "discrete" if hasattr(act_space, "n") else "continuous",
        }

    def create_agent(self, algorithm: str, params: dict | None = None) -> dict:
        """Create an RL agent."""
        import stable_baselines3 as sb3

        if algorithm not in ALGORITHMS:
            return {"status": "error", "message": f"Unknown algorithm: {algorithm}"}

        algo_cls = getattr(sb3, algorithm, None)
        if not algo_cls:
            return {"status": "error", "message": f"SB3 does not have: {algorithm}"}

        self.algorithm = algorithm
        algo_params = {}
        defaults = ALGORITHMS[algorithm]["params"]

        # Apply defaults then overrides
        for key, info in defaults.items():
            val = (params or {}).get(key, info["default"])
            if info["type"] == "float" and val != "auto":
                val = float(val)
            elif info["type"] == "int":
                val = int(val)
            algo_params[key] = val

        env = self.env
        if self._custom_env:
            # Wrap custom env for SB3
            env = self._wrap_custom_env()

        self.model = algo_cls("MlpPolicy", env, verbose=0, **algo_params)

        total_params = sum(p.numel() for p in self.model.policy.parameters())
        return {
            "status": "created",
            "algorithm": algorithm,
            "total_params": total_params,
            "params": algo_params,
        }

    def _wrap_custom_env(self):
        """Wrap CustomGridEnv in Gymnasium-compatible interface."""
        import gymnasium as gym
        import numpy as np

        env = self._custom_env

        class WrappedGridEnv(gym.Env):
            def __init__(self, grid_env):
                self.grid = grid_env
                self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
                self.action_space = gym.spaces.Discrete(4)

            def reset(self, **kwargs):
                obs = self.grid.reset()
                return np.array(obs, dtype=np.float32), {}

            def step(self, action):
                obs, reward, done, info = self.grid.step(action)
                return np.array(obs, dtype=np.float32), reward, done, False, info

        return WrappedGridEnv(env)

    def start_training(self, total_timesteps: int = 100000, eval_freq: int = 1000) -> dict:
        """Start RL training in background thread."""
        if self._is_training:
            return {"status": "already_training"}
        if self.model is None:
            return {"status": "error", "message": "Create agent first"}

        self._stop_event.clear()
        self._is_training = True
        self._history = []

        self._train_thread = threading.Thread(
            target=self._train_loop,
            args=(total_timesteps, eval_freq),
            daemon=True,
        )
        self._train_thread.start()
        return {"status": "started", "total_timesteps": total_timesteps}

    def stop_training(self) -> dict:
        if not self._is_training:
            return {"status": "not_training"}
        self._stop_event.set()
        if self._train_thread:
            self._train_thread.join(timeout=10)
        self._is_training = False
        return {"status": "stopped"}

    def _train_loop(self, total_timesteps: int, eval_freq: int) -> None:
        """Training loop with metric broadcasting."""
        try:
            from stable_baselines3.common.callbacks import BaseCallback

            engine = self

            class MetricsCallback(BaseCallback):
                def __init__(self):
                    super().__init__()
                    self._ep_rewards = []
                    self._ep_lengths = []

                def _on_step(self) -> bool:
                    if engine._stop_event.is_set():
                        return False

                    # Collect episode info
                    infos = self.locals.get("infos", [])
                    for info in infos:
                        if "episode" in info:
                            ep_reward = info["episode"]["r"]
                            ep_length = info["episode"]["l"]
                            self._ep_rewards.append(ep_reward)
                            self._ep_lengths.append(ep_length)

                            step_data = {
                                "timestep": self.num_timesteps,
                                "episode_reward": ep_reward,
                                "episode_length": ep_length,
                                "mean_reward_100": sum(self._ep_rewards[-100:]) / max(len(self._ep_rewards[-100:]), 1),
                                "total_episodes": len(self._ep_rewards),
                            }
                            engine._history.append(step_data)
                            engine._emit("rl_step", step_data)

                    # Emit progress periodically
                    if self.num_timesteps % eval_freq == 0:
                        engine._emit("rl_progress", {
                            "timestep": self.num_timesteps,
                            "total": total_timesteps,
                            "pct": round(self.num_timesteps / total_timesteps * 100, 1),
                        })

                    return True

            callback = MetricsCallback()

            self._emit("rl_training_status", {"status": "started"})

            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=False,
            )

            self._emit("rl_training_complete", {
                "total_episodes": len(self._history),
                "final_mean_reward": self._history[-1]["mean_reward_100"] if self._history else 0,
            })

        except Exception as e:
            self._emit("error", {"message": str(e), "traceback": traceback.format_exc()})
        finally:
            self._is_training = False

    def run_episode(self, render_steps: bool = True) -> dict:
        """Run a single episode and return trajectory for visualization."""
        if self.model is None:
            return {"status": "error", "message": "No trained model"}

        frames = []
        env = self.env
        is_custom = self._custom_env is not None

        if is_custom:
            obs = self._custom_env.reset()
            done = False
            total_reward = 0
            step = 0
            while not done:
                import numpy as np
                obs_tensor = np.array(obs, dtype=np.float32)
                action, _ = self.model.predict(obs_tensor, deterministic=True)
                obs, reward, done, info = self._custom_env.step(int(action))
                total_reward += reward
                step += 1
                if render_steps:
                    state = self._custom_env.get_state_for_render()
                    state["step"] = step
                    state["reward"] = total_reward
                    state["action"] = int(action)
                    frames.append(state)
        else:
            obs, _ = env.reset()
            done = False
            total_reward = 0
            step = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1
                if render_steps:
                    frames.append({
                        "step": step,
                        "obs": obs.tolist() if hasattr(obs, "tolist") else list(obs),
                        "action": action.tolist() if hasattr(action, "tolist") else int(action),
                        "reward": float(total_reward),
                    })

        self._episode_frames = frames
        return {
            "status": "ok",
            "total_reward": total_reward,
            "steps": step,
            "frames": frames[:500],  # Cap for transfer
        }

    def save_model(self, path: str = "./sg_outputs/rl_model") -> dict:
        if self.model is None:
            return {"status": "error", "message": "No model"}
        self.model.save(path)
        return {"status": "saved", "path": path}

    def load_model(self, path: str, algorithm: str) -> dict:
        import stable_baselines3 as sb3
        algo_cls = getattr(sb3, algorithm)
        env = self.env or (self._wrap_custom_env() if self._custom_env else None)
        self.model = algo_cls.load(path, env=env)
        self.algorithm = algorithm
        return {"status": "loaded", "path": path}

    def get_info(self) -> dict:
        return {
            "env_id": self.env_id,
            "algorithm": self.algorithm,
            "is_training": self._is_training,
            "total_episodes": len(self._history),
            "history_length": len(self._history),
            "has_model": self.model is not None,
        }

    def get_history(self) -> list[dict]:
        return self._history
