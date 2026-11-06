"""Session search (SDK >= 0.38 `sessions/search`), as the SDK documents it:
session_id exact, display_name case-insensitive with every word present (either one
matching when both are given), user_metadata every pair exact, statuses any-of,
newest first, a (created_at, session_id) cursor. Pure: the route adds checkpoint counts."""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..models.requests import DEFAULT_PROJECT, LOCAL_OWNER, SessionSearchRequest, SessionStatusQuery
from .sessions import Session


class UnknownProject(ValueError):
    """A project id other than the one this server has (a 404, as Tinker answers an unknown project)."""


def status_of(session: Session) -> Dict[str, Any]:
    """The SDK SessionStatus of a stored session. An expired session is reaped from the
    store, so a stored one has either finished from its client or is still active."""
    if session.finished_at is None:
        return {"type": "active"}
    return {"type": "client_finished", "reason": {"type": session.finish_reason}, "detail": session.finish_detail}


def _status_matches(status: Dict[str, Any], queries: Optional[List[SessionStatusQuery]]) -> bool:
    if queries is None:
        return True
    for q in queries:
        if q.type != status["type"]:
            continue
        if q.type == "client_finished" and q.reasons is not None and status["reason"]["type"] not in q.reasons:
            continue
        return True
    return False


def _name_matches(name: Optional[str], query: str) -> bool:
    return name is not None and all(word in name.lower() for word in query.lower().split())


def _identity_matches(s: Session, req: SessionSearchRequest) -> bool:
    by_id = req.session_id is not None and s.session_id == req.session_id
    by_name = req.display_name is not None and _name_matches(s.display_name, req.display_name)
    if req.session_id is None and req.display_name is None:
        return True
    return by_id or by_name


def _metadata_matches(s: Session, wanted: Dict[str, str]) -> bool:
    return all(k in s.user_metadata and str(s.user_metadata[k]) == v for k, v in wanted.items())


def _after_cursor(s: Session, req: SessionSearchRequest) -> bool:
    if req.cursor is None:
        return True
    key = (datetime.fromisoformat(s.created_at), s.session_id)
    return key < (req.cursor.last_created_at, req.cursor.last_session_id)


def search(sessions: List[Session], req: SessionSearchRequest) -> Tuple[List[Session], Optional[Dict[str, str]]]:
    """`sessions` newest first; returns one page and the next cursor (None once exhausted).
    Raises UnknownProject for a project id this server does not have."""
    unknown = [p for p in (req.project_ids or []) if p != DEFAULT_PROJECT]
    if unknown:
        raise UnknownProject(f"project {', '.join(unknown)} not found; this server has one project, {DEFAULT_PROJECT!r}")
    if req.owner.type == "user":
        assert req.owner.owner_urns is not None  # SearchOwner's validator pairs them
        if LOCAL_OWNER not in req.owner.owner_urns:
            return [], None
    matches = [s for s in sessions
               if _after_cursor(s, req) and _identity_matches(s, req) and _metadata_matches(s, req.user_metadata)
               and _status_matches(status_of(s), req.statuses)]
    page = matches[:req.limit]
    if len(matches) <= req.limit:
        return page, None
    last = page[-1]
    return page, {"last_created_at": last.created_at, "last_session_id": last.session_id}
