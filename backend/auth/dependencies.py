from __future__ import annotations

from fastapi import HTTPException, Request


def get_current_user(request: Request) -> dict | None:
    """Return the user dict from the session, or ``None``."""
    return request.session.get("user")


def require_user(request: Request) -> dict:
    """Raise 401 if no user is logged in."""
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_admin(request: Request) -> dict:
    """Raise 401 if not logged in, 403 if not admin."""
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
