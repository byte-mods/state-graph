"""Server-side physics engine — MuJoCo for sub-millisecond accurate simulation.

Runs physics at 500Hz+ server-side, streams state to browser at 60fps.
Falls back to simplified physics if MuJoCo not installed.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable


class PhysicsServer:
    """High-fidelity physics running server-side, streamed to browser."""

    def __init__(self):
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._broadcast: Callable | None = None
        self._loop = None
        self._timestep = 0.002  # 500Hz (MuJoCo default)
        self._sim_time = 0.0
        self._state: dict = {}
        self._bodies: list[dict] = []
        self._joints: list[dict] = []
        self._gravity = [0, -9.81, 0]
        self._use_mujoco = False
        self._mujoco_model = None
        self._mujoco_data = None

    def set_broadcast(self, fn: Callable, loop) -> None:
        self._broadcast = fn
        self._loop = loop

    def _emit(self, event: str, data: dict) -> None:
        if self._broadcast and self._loop:
            import asyncio
            asyncio.run_coroutine_threadsafe(self._broadcast(event, data), self._loop)

    def load_scene(self, bodies: list[dict], joints: list[dict] = None, use_mujoco: bool = True) -> dict:
        """Load a scene with rigid bodies and joints."""
        self._bodies = bodies
        self._joints = joints or []
        self._sim_time = 0.0

        if use_mujoco:
            try:
                return self._load_mujoco(bodies, joints or [])
            except ImportError:
                pass

        # Fallback: simple Euler integration physics
        self._use_mujoco = False
        for b in self._bodies:
            b.setdefault("velocity", [0, 0, 0])
            b.setdefault("angular_velocity", [0, 0, 0])
            b.setdefault("force", [0, 0, 0])
            b.setdefault("fixed", b.get("mass", 0) == 0)

        return {
            "status": "loaded",
            "engine": "euler_fallback",
            "bodies": len(self._bodies),
            "joints": len(self._joints),
            "timestep": self._timestep,
        }

    def _load_mujoco(self, bodies: list[dict], joints: list[dict]) -> dict:
        """Build and load a MuJoCo model from body/joint descriptions."""
        import mujoco

        # Generate MJCF XML
        xml = self._generate_mjcf(bodies, joints)
        self._mujoco_model = mujoco.MjModel.from_xml_string(xml)
        self._mujoco_data = mujoco.MjData(self._mujoco_model)
        self._use_mujoco = True
        self._timestep = self._mujoco_model.opt.timestep

        return {
            "status": "loaded",
            "engine": "mujoco",
            "bodies": self._mujoco_model.nbody,
            "joints": self._mujoco_model.njnt,
            "timestep": self._timestep,
            "nq": self._mujoco_model.nq,
            "nv": self._mujoco_model.nv,
        }

    def _generate_mjcf(self, bodies: list[dict], joints: list[dict]) -> str:
        """Generate MuJoCo MJCF XML from body descriptions."""
        xml = '<mujoco model="sg_robot">\n'
        xml += '  <option timestep="0.002" gravity="0 0 -9.81"/>\n'
        xml += '  <worldbody>\n'
        xml += '    <light pos="0 0 3" dir="0 0 -1"/>\n'
        xml += '    <geom type="plane" size="5 5 0.1" rgba="0.2 0.2 0.3 1"/>\n'

        for i, b in enumerate(bodies):
            pos = b.get("position", [0, 0, 0])
            dim = b.get("dimensions", {"x": 0.05, "y": 0.05, "z": 0.05})
            mass = b.get("mass", 0.1)
            name = b.get("name", f"body_{i}")

            xml += f'    <body name="{name}" pos="{pos[0]} {pos[2]} {pos[1]}">\n'
            if not b.get("fixed", False):
                xml += f'      <joint type="free"/>\n'
            xml += f'      <geom type="box" size="{dim["x"]/2} {dim["z"]/2} {dim["y"]/2}" mass="{mass}" rgba="0.5 0.5 0.8 1"/>\n'
            xml += f'    </body>\n'

        xml += '  </worldbody>\n'
        xml += '</mujoco>\n'
        return xml

    def start(self) -> dict:
        """Start physics simulation loop."""
        if self._running:
            return {"status": "already_running"}
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._physics_loop, daemon=True)
        self._thread.start()
        return {"status": "started", "engine": "mujoco" if self._use_mujoco else "euler"}

    def stop(self) -> dict:
        self._stop_event.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return {"status": "stopped"}

    def step_once(self) -> dict:
        """Run a single physics step."""
        if self._use_mujoco:
            return self._step_mujoco()
        return self._step_euler()

    def _physics_loop(self) -> None:
        """Main physics loop — runs at physics rate, broadcasts at 60fps."""
        broadcast_interval = 1.0 / 60  # 60fps to browser
        last_broadcast = 0
        steps_per_broadcast = max(1, int(broadcast_interval / self._timestep))

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            # Run physics steps
            for _ in range(steps_per_broadcast):
                if self._stop_event.is_set():
                    break
                if self._use_mujoco:
                    self._step_mujoco()
                else:
                    self._step_euler()
                self._sim_time += self._timestep

            # Broadcast state at 60fps
            now = time.perf_counter()
            if now - last_broadcast >= broadcast_interval:
                state = self._get_state()
                state["sim_time"] = round(self._sim_time, 4)
                self._emit("physics_state", state)
                last_broadcast = now

            # Sleep to maintain real-time
            elapsed = time.perf_counter() - t0
            target = steps_per_broadcast * self._timestep
            if elapsed < target:
                time.sleep(target - elapsed)

    def _step_mujoco(self) -> dict:
        import mujoco
        mujoco.mj_step(self._mujoco_model, self._mujoco_data)
        return {"status": "ok", "time": self._mujoco_data.time}

    def _step_euler(self) -> dict:
        """Simple Euler integration fallback."""
        dt = self._timestep
        g = self._gravity

        for b in self._bodies:
            if b.get("fixed"):
                continue

            mass = b.get("mass", 1.0)
            pos = b["position"]
            vel = b["velocity"]

            # Gravity + applied forces
            fx = b["force"][0] + 0
            fy = b["force"][1] + g[1] * mass
            fz = b["force"][2] + 0

            # Euler integration
            vel[0] += (fx / mass) * dt
            vel[1] += (fy / mass) * dt
            vel[2] += (fz / mass) * dt

            # Damping
            vel[0] *= 0.999
            vel[1] *= 0.999
            vel[2] *= 0.999

            pos[0] += vel[0] * dt
            pos[1] += vel[1] * dt
            pos[2] += vel[2] * dt

            # Ground collision
            half_h = b.get("dimensions", {}).get("y", 0.05) / 2
            if pos[1] < half_h:
                pos[1] = half_h
                vel[1] = -vel[1] * 0.3  # Bounce with energy loss
                # Friction
                vel[0] *= 0.95
                vel[2] *= 0.95

        return {"status": "ok", "time": self._sim_time}

    def _get_state(self) -> dict:
        """Get current physics state for browser rendering."""
        if self._use_mujoco and self._mujoco_data:
            import mujoco
            bodies = []
            for i in range(self._mujoco_model.nbody):
                pos = self._mujoco_data.xpos[i].tolist()
                quat = self._mujoco_data.xquat[i].tolist()
                bodies.append({
                    "id": i,
                    "position": [pos[0], pos[2], pos[1]],  # MuJoCo Z-up → Y-up
                    "quaternion": quat,
                })
            return {"bodies": bodies}

        # Euler fallback
        return {
            "bodies": [
                {
                    "id": i,
                    "position": b["position"],
                    "velocity": b["velocity"],
                }
                for i, b in enumerate(self._bodies) if not b.get("fixed")
            ]
        }

    def apply_force(self, body_index: int, force: list[float]) -> dict:
        """Apply external force to a body."""
        if self._use_mujoco:
            self._mujoco_data.xfrc_applied[body_index] = force + [0, 0, 0]
            return {"status": "ok"}
        if body_index < len(self._bodies):
            self._bodies[body_index]["force"] = force
            return {"status": "ok"}
        return {"status": "error"}

    def set_joint_target(self, joint_index: int, target: float) -> dict:
        """Set target angle for a joint (MuJoCo actuator)."""
        if self._use_mujoco and self._mujoco_data:
            if joint_index < self._mujoco_model.nu:
                self._mujoco_data.ctrl[joint_index] = target
                return {"status": "ok"}
        return {"status": "error", "message": "Joint not found or MuJoCo not loaded"}

    def get_info(self) -> dict:
        return {
            "running": self._running,
            "engine": "mujoco" if self._use_mujoco else "euler",
            "timestep": self._timestep,
            "sim_time": round(self._sim_time, 4),
            "body_count": len(self._bodies),
            "fps_target": int(1.0 / self._timestep),
        }
