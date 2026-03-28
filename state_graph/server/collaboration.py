"""Multi-user collaboration — sessions, rooms, shared state, live cursors."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


class User:
    """A connected user session."""

    def __init__(self, name: str, color: str | None = None):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.color = color or COLORS[hash(name) % len(COLORS)]
        self.cursor: dict | None = None  # {x, y, panel}
        self.active_file: str | None = None
        self.connected_at = time.time()
        self.last_seen = time.time()


class Room:
    """A collaboration room — shared project workspace."""

    def __init__(self, name: str, project_id: str | None = None):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.project_id = project_id
        self.users: dict[str, User] = {}
        self.created_at = time.time()
        self.chat_history: list[dict] = []
        self._locks: dict[str, str] = {}  # file_path -> user_id

    def add_user(self, user: User) -> dict:
        self.users[user.id] = user
        return {"status": "joined", "user": self._user_dict(user), "room_id": self.id}

    def remove_user(self, user_id: str) -> dict:
        user = self.users.pop(user_id, None)
        # Release locks held by this user
        self._locks = {k: v for k, v in self._locks.items() if v != user_id}
        return {"status": "left", "user_id": user_id}

    def update_cursor(self, user_id: str, cursor: dict) -> dict:
        user = self.users.get(user_id)
        if user:
            user.cursor = cursor
            user.last_seen = time.time()
        return {"cursors": self._all_cursors()}

    def lock_file(self, user_id: str, file_path: str) -> dict:
        current_lock = self._locks.get(file_path)
        if current_lock and current_lock != user_id:
            locker = self.users.get(current_lock)
            return {"status": "locked", "by": locker.name if locker else "unknown"}
        self._locks[file_path] = user_id
        return {"status": "acquired", "file": file_path}

    def unlock_file(self, user_id: str, file_path: str) -> dict:
        if self._locks.get(file_path) == user_id:
            del self._locks[file_path]
        return {"status": "released", "file": file_path}

    def send_chat(self, user_id: str, message: str) -> dict:
        user = self.users.get(user_id)
        if not user:
            return {"status": "error"}
        entry = {
            "id": str(uuid.uuid4())[:6],
            "user_id": user_id,
            "user_name": user.name,
            "user_color": user.color,
            "message": message,
            "timestamp": time.time(),
        }
        self.chat_history.append(entry)
        if len(self.chat_history) > 500:
            self.chat_history = self.chat_history[-500:]
        return {"status": "sent", "chat": entry}

    def get_state(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "users": [self._user_dict(u) for u in self.users.values()],
            "locks": self._locks,
            "chat_count": len(self.chat_history),
        }

    def _all_cursors(self) -> list[dict]:
        return [
            {"user_id": u.id, "name": u.name, "color": u.color, "cursor": u.cursor}
            for u in self.users.values() if u.cursor
        ]

    def _user_dict(self, u: User) -> dict:
        return {"id": u.id, "name": u.name, "color": u.color, "active_file": u.active_file}


class CollaborationManager:
    """Manages rooms and user sessions."""

    def __init__(self):
        self.rooms: dict[str, Room] = {}
        self.users: dict[str, User] = {}  # All connected users
        self._user_rooms: dict[str, str] = {}  # user_id -> room_id

    def create_user(self, name: str) -> User:
        user = User(name)
        self.users[user.id] = user
        return user

    def create_room(self, name: str, project_id: str | None = None) -> Room:
        room = Room(name, project_id)
        self.rooms[room.id] = room
        return room

    def join_room(self, user_id: str, room_id: str) -> dict:
        user = self.users.get(user_id)
        room = self.rooms.get(room_id)
        if not user or not room:
            return {"status": "error", "message": "User or room not found"}
        # Leave current room if in one
        if user_id in self._user_rooms:
            old_room = self.rooms.get(self._user_rooms[user_id])
            if old_room:
                old_room.remove_user(user_id)
        self._user_rooms[user_id] = room_id
        return room.add_user(user)

    def leave_room(self, user_id: str) -> dict:
        room_id = self._user_rooms.pop(user_id, None)
        if room_id:
            room = self.rooms.get(room_id)
            if room:
                return room.remove_user(user_id)
        return {"status": "ok"}

    def get_user_room(self, user_id: str) -> Room | None:
        room_id = self._user_rooms.get(user_id)
        return self.rooms.get(room_id) if room_id else None

    def list_rooms(self) -> list[dict]:
        return [r.get_state() for r in self.rooms.values()]

    def delete_room(self, room_id: str) -> dict:
        room = self.rooms.pop(room_id, None)
        if room:
            for uid in list(room.users.keys()):
                self._user_rooms.pop(uid, None)
            return {"status": "deleted"}
        return {"status": "error"}


COLORS = [
    "#7c6cf0", "#f06292", "#4cceac", "#f0c040", "#00d4c8",
    "#f09040", "#74b9ff", "#a29bfe", "#fd79a8", "#55efc4",
    "#e17055", "#0984e3", "#00b894", "#e84393", "#fab1a0",
]
