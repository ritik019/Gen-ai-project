from __future__ import annotations

from typing import Any

import bcrypt

_users: dict[str, dict[str, Any]] = {}


def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _seed_users() -> None:
    """Pre-seed demo users on import."""
    _users["user"] = {"password_hash": _hash_password("user123"), "role": "user"}
    _users["admin"] = {"password_hash": _hash_password("admin123"), "role": "admin"}


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    """Verify credentials. Returns ``{username, role}`` or ``None``."""
    record = _users.get(username)
    if record and _verify_password(password, record["password_hash"]):
        return {"username": username, "role": record["role"]}
    return None


_seed_users()
