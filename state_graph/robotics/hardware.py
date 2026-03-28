"""Hardware-in-the-loop bridge — Serial/USB communication with physical robots.

Supports: Arduino, ESP32, Raspberry Pi, any serial device.
Protocol: JSON messages over serial for joint commands and sensor reads.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable


class HardwareBridge:
    """Bidirectional serial bridge to physical robot hardware."""

    def __init__(self):
        self._port: str = ""
        self._baud: int = 115200
        self._serial = None
        self._is_connected = False
        self._read_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._broadcast: Callable | None = None
        self._loop = None
        self._sensor_data: dict = {}
        self._last_command: dict = {}

    def set_broadcast(self, fn: Callable, loop) -> None:
        self._broadcast = fn
        self._loop = loop

    def _emit(self, event: str, data: dict) -> None:
        if self._broadcast and self._loop:
            import asyncio
            asyncio.run_coroutine_threadsafe(self._broadcast(event, data), self._loop)

    def list_ports(self) -> list[dict]:
        """List available serial ports."""
        try:
            import serial.tools.list_ports
            ports = []
            for port in serial.tools.list_ports.comports():
                ports.append({
                    "device": port.device,
                    "description": port.description,
                    "hwid": port.hwid,
                    "manufacturer": port.manufacturer or "",
                })
            return ports
        except ImportError:
            return [{"error": "pyserial not installed. Run: pip install pyserial"}]

    def connect(self, port: str, baud: int = 115200, timeout: float = 2.0) -> dict:
        """Connect to a serial device."""
        try:
            import serial
            self._serial = serial.Serial(port, baud, timeout=timeout)
            self._port = port
            self._baud = baud
            self._is_connected = True

            # Start read thread
            self._stop_event.clear()
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()

            time.sleep(0.5)  # Wait for Arduino reset
            return {"status": "connected", "port": port, "baud": baud}
        except ImportError:
            return {"status": "error", "message": "pyserial not installed. Run: pip install pyserial"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def disconnect(self) -> dict:
        self._stop_event.set()
        self._is_connected = False
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        return {"status": "disconnected"}

    def send_command(self, command: dict) -> dict:
        """Send a JSON command to the hardware."""
        if not self._is_connected or not self._serial:
            return {"status": "error", "message": "Not connected"}
        try:
            msg = json.dumps(command) + "\n"
            self._serial.write(msg.encode())
            self._last_command = command
            return {"status": "sent", "command": command}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def send_joint_angles(self, angles: dict[str, float]) -> dict:
        """Send joint angle commands. Format: {joint_name: angle_degrees}."""
        return self.send_command({"type": "joints", "angles": angles})

    def send_motor_speeds(self, speeds: dict[str, float]) -> dict:
        """Send motor speed commands. Format: {motor_name: speed_pct (-100 to 100)}."""
        return self.send_command({"type": "motors", "speeds": speeds})

    def send_raw(self, data: str) -> dict:
        """Send raw string data."""
        if not self._is_connected or not self._serial:
            return {"status": "error", "message": "Not connected"}
        try:
            self._serial.write((data + "\n").encode())
            return {"status": "sent"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_sensor_data(self) -> dict:
        """Get latest sensor readings from hardware."""
        return {"status": "ok", "sensors": self._sensor_data, "connected": self._is_connected}

    def _read_loop(self) -> None:
        """Background thread reading serial data."""
        while not self._stop_event.is_set():
            try:
                if self._serial and self._serial.in_waiting:
                    line = self._serial.readline().decode("utf-8", errors="replace").strip()
                    if line:
                        try:
                            data = json.loads(line)
                            self._sensor_data = data
                            self._emit("hardware_sensor", data)
                        except json.JSONDecodeError:
                            self._emit("hardware_raw", {"data": line})
            except Exception:
                pass
            time.sleep(0.01)  # 100Hz polling

    def upload_firmware(self, board: str, sketch_path: str) -> dict:
        """Compile and upload Arduino firmware (requires arduino-cli)."""
        import subprocess
        try:
            result = subprocess.run(
                ["arduino-cli", "compile", "--fqbn", board, sketch_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                return {"status": "error", "stage": "compile", "stderr": result.stderr}

            result = subprocess.run(
                ["arduino-cli", "upload", "-p", self._port, "--fqbn", board, sketch_path],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                return {"status": "error", "stage": "upload", "stderr": result.stderr}

            return {"status": "uploaded", "board": board}
        except FileNotFoundError:
            return {"status": "error", "message": "arduino-cli not found. Install from https://arduino.github.io/arduino-cli/"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def generate_firmware(self, robot_config: dict) -> str:
        """Generate Arduino sketch for a robot configuration."""
        servos = [c for c in robot_config.get("components", []) if "servo" in c.get("type", "")]
        motors = [c for c in robot_config.get("components", []) if "motor" in c.get("type", "") and "servo" not in c.get("type", "")]

        code = '#include <Servo.h>\n#include <ArduinoJson.h>\n\n'

        # Servo declarations
        for i, s in enumerate(servos):
            code += f'Servo servo_{i}; // {s.get("role", f"joint_{i}")}\n'

        code += f'\nconst int NUM_SERVOS = {len(servos)};\n'
        code += f'const int NUM_MOTORS = {len(motors)};\n'
        code += 'int servo_pins[] = {' + ', '.join(str(3 + i) for i in range(len(servos))) + '};\n'
        code += 'float servo_angles[' + str(max(len(servos), 1)) + '];\n\n'

        code += 'void setup() {\n'
        code += '  Serial.begin(115200);\n'
        for i in range(len(servos)):
            code += f'  servo_{i}.attach(servo_pins[{i}]);\n'
        code += '}\n\n'

        code += 'void loop() {\n'
        code += '  if (Serial.available()) {\n'
        code += '    String json = Serial.readStringUntil(\'\\n\');\n'
        code += '    StaticJsonDocument<512> doc;\n'
        code += '    DeserializationError err = deserializeJson(doc, json);\n'
        code += '    if (!err && doc["type"] == "joints") {\n'
        code += '      JsonObject angles = doc["angles"];\n'
        for i, s in enumerate(servos):
            role = s.get("role", f"joint_{i}")
            code += f'      if (angles.containsKey("{role}")) servo_{i}.write(angles["{role}"].as<int>());\n'
        code += '    }\n'
        code += '  }\n\n'

        # Send sensor data back
        code += '  // Send sensor data every 50ms\n'
        code += '  static unsigned long lastSend = 0;\n'
        code += '  if (millis() - lastSend > 50) {\n'
        code += '    StaticJsonDocument<256> out;\n'
        code += '    out["type"] = "sensors";\n'
        code += '    out["uptime"] = millis();\n'
        code += '    // Add your sensor readings here\n'
        code += '    // out["distance"] = analogRead(A0);\n'
        code += '    serializeJson(out, Serial);\n'
        code += '    Serial.println();\n'
        code += '    lastSend = millis();\n'
        code += '  }\n'
        code += '}\n'

        return code

    def get_info(self) -> dict:
        return {
            "connected": self._is_connected,
            "port": self._port,
            "baud": self._baud,
            "last_sensor": self._sensor_data,
            "last_command": self._last_command,
        }
