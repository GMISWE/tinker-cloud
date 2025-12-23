"""
Session storage module for managing sessions and samplers.

This module provides a SQLite-backed storage layer for tracking sessions,
samplers, and session-model relationships with thread-safe access.
"""
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SessionStorage:
    """
    SQLite-backed storage for sessions and samplers.

    This class provides thread-safe storage for tracking sessions,
    their associated models, and sampling sessions with persistence.
    """

    def __init__(self, db_path: Path):
        """
        Initialize session storage with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._lock = Lock()
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database schema with migration guard."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                # Sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        sdk_version TEXT,
                        tags TEXT,
                        user_metadata TEXT,
                        created_at TEXT NOT NULL,
                        last_heartbeat TEXT NOT NULL
                    )
                """)

                # Samplers table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS samplers (
                        sampler_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        model_id TEXT,
                        base_model TEXT,
                        model_path TEXT,
                        created_at TEXT NOT NULL
                    )
                """)

                # Session models table (track model_seq_id + context for matching)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_models (
                        session_id TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        model_seq_id INTEGER,
                        base_model TEXT,
                        model_path TEXT,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (session_id, model_id)
                    )
                """)

                # Create indices for common queries
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_samplers_session ON samplers(session_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_heartbeat ON sessions(last_heartbeat)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_models_seq ON session_models(session_id, model_seq_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_models_model ON session_models(model_id)"
                )

                conn.commit()
                logger.info(f"Initialized session database at {self.db_path}")
            finally:
                conn.close()

    # =========================================================================
    # Session CRUD
    # =========================================================================

    def save_session(
        self,
        session_id: str,
        sdk_version: str,
        tags: List[str],
        user_metadata: Dict[str, Any],
        created_at: datetime,
        last_heartbeat: datetime
    ) -> bool:
        """
        Save a new session to storage.

        Args:
            session_id: Unique session identifier
            sdk_version: Client SDK version
            tags: Client-provided tags
            user_metadata: Client-provided metadata
            created_at: Session creation time
            last_heartbeat: Last heartbeat time

        Returns:
            True if saved successfully
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions
                    (session_id, sdk_version, tags, user_metadata, created_at, last_heartbeat)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    sdk_version,
                    json.dumps(tags),
                    json.dumps(user_metadata),
                    created_at.isoformat(),
                    last_heartbeat.isoformat()
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to save session {session_id}: {e}")
                raise
            finally:
                conn.close()

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a session from storage.

        Args:
            session_id: Session identifier

        Returns:
            Session data dict or None if not found
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id, sdk_version, tags, user_metadata, created_at, last_heartbeat
                    FROM sessions
                    WHERE session_id = ?
                """, (session_id,))
                row = cursor.fetchone()

                if row:
                    return {
                        "session_id": row[0],
                        "sdk_version": row[1],
                        "tags": json.loads(row[2]) if row[2] else [],
                        "user_metadata": json.loads(row[3]) if row[3] else {},
                        "created_at": row[4],
                        "last_heartbeat": row[5]
                    }
                return None
            finally:
                conn.close()

    def list_sessions(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List sessions with pagination.

        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            List of session data dicts
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id, sdk_version, tags, user_metadata, created_at, last_heartbeat
                    FROM sessions
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))

                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        "session_id": row[0],
                        "sdk_version": row[1],
                        "tags": json.loads(row[2]) if row[2] else [],
                        "user_metadata": json.loads(row[3]) if row[3] else {},
                        "created_at": row[4],
                        "last_heartbeat": row[5]
                    })
                return sessions
            finally:
                conn.close()

    def update_heartbeat(self, session_id: str) -> bool:
        """
        Update heartbeat timestamp for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if update successful, False if session not found
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions
                    SET last_heartbeat = ?
                    WHERE session_id = ?
                """, (datetime.utcnow().isoformat(), session_id))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its related data.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                # Delete associated samplers
                cursor.execute("DELETE FROM samplers WHERE session_id = ?", (session_id,))

                # Delete associated models
                cursor.execute("DELETE FROM session_models WHERE session_id = ?", (session_id,))

                # Delete session
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.

        Args:
            session_id: Session identifier

        Returns:
            True if session exists
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?",
                    (session_id,)
                )
                return cursor.fetchone() is not None
            finally:
                conn.close()

    # =========================================================================
    # Sampler CRUD
    # =========================================================================

    def save_sampler(
        self,
        sampler_id: str,
        session_id: str,
        model_id: Optional[str],
        base_model: Optional[str],
        model_path: Optional[str]
    ) -> bool:
        """
        Save a sampler to storage.

        Args:
            sampler_id: Unique sampler identifier (sampling_session_id)
            session_id: Parent session identifier
            model_id: Model that created this sampler (optional)
            base_model: Base model name (optional, None if unknown)
            model_path: Model path/checkpoint path (optional)

        Returns:
            True if saved successfully
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO samplers
                    (sampler_id, session_id, model_id, base_model, model_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    sampler_id,
                    session_id,
                    model_id,
                    base_model,
                    model_path,
                    datetime.utcnow().isoformat()
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to save sampler {sampler_id}: {e}")
                raise
            finally:
                conn.close()

    def load_sampler(self, sampler_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a sampler from storage.

        Args:
            sampler_id: Sampler identifier

        Returns:
            Sampler data dict or None if not found
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT sampler_id, session_id, model_id, base_model, model_path, created_at
                    FROM samplers
                    WHERE sampler_id = ?
                """, (sampler_id,))
                row = cursor.fetchone()

                if row:
                    return {
                        "sampler_id": row[0],
                        "session_id": row[1],
                        "model_id": row[2],
                        "base_model": row[3],
                        "model_path": row[4],
                        "created_at": row[5]
                    }
                return None
            finally:
                conn.close()

    def list_samplers_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        List all samplers for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of sampler data dicts
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT sampler_id, session_id, model_id, base_model, model_path, created_at
                    FROM samplers
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                """, (session_id,))

                samplers = []
                for row in cursor.fetchall():
                    samplers.append({
                        "sampler_id": row[0],
                        "session_id": row[1],
                        "model_id": row[2],
                        "base_model": row[3],
                        "model_path": row[4],
                        "created_at": row[5]
                    })
                return samplers
            finally:
                conn.close()

    # =========================================================================
    # Model Tracking
    # =========================================================================

    def add_model_to_session(
        self,
        session_id: str,
        model_id: str,
        model_seq_id: int,
        base_model: Optional[str],
        model_path: Optional[str]
    ) -> bool:
        """
        Link a model to a session with context for matching.

        Args:
            session_id: Session identifier
            model_id: Model identifier
            model_seq_id: Sequence ID for ordering within session
            base_model: Base model name (for context matching)
            model_path: Model/checkpoint path (for context matching)

        Returns:
            True if saved successfully
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO session_models
                    (session_id, model_id, model_seq_id, base_model, model_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    model_id,
                    model_seq_id,
                    base_model,
                    model_path,
                    datetime.utcnow().isoformat()
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to add model {model_id} to session {session_id}: {e}")
                raise
            finally:
                conn.close()

    def remove_model_from_session(self, model_id: str) -> Optional[str]:
        """
        Remove a model from its session.

        Args:
            model_id: Model identifier

        Returns:
            Session ID that owned the model, or None if not found
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                # First get the session_id before deleting
                cursor.execute(
                    "SELECT session_id FROM session_models WHERE model_id = ?",
                    (model_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return None

                session_id = row[0]

                # Delete the model
                cursor.execute(
                    "DELETE FROM session_models WHERE model_id = ?",
                    (model_id,)
                )
                conn.commit()

                return session_id
            finally:
                conn.close()

    def get_models_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all models for a session, ordered by model_seq_id.

        Args:
            session_id: Session identifier

        Returns:
            List of model data dicts, sorted by model_seq_id
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT model_id, model_seq_id, base_model, model_path, created_at
                    FROM session_models
                    WHERE session_id = ?
                    ORDER BY model_seq_id ASC
                """, (session_id,))

                models = []
                for row in cursor.fetchall():
                    models.append({
                        "model_id": row[0],
                        "model_seq_id": row[1],
                        "base_model": row[2],
                        "model_path": row[3],
                        "created_at": row[4]
                    })
                return models
            finally:
                conn.close()

    def get_model_context(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model context including session_id for discovery.

        Args:
            model_id: Model identifier

        Returns:
            Dict with session_id, base_model, model_path, or None if not found
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id, base_model, model_path
                    FROM session_models
                    WHERE model_id = ?
                """, (model_id,))
                row = cursor.fetchone()

                if row:
                    return {
                        "session_id": row[0],
                        "base_model": row[1],
                        "model_path": row[2]
                    }
                return None
            finally:
                conn.close()

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup_stale_sessions(self, max_age_hours: int = 24) -> Tuple[int, List[str]]:
        """
        Delete sessions with heartbeat older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Tuple of (count_removed, list_of_removed_session_ids)

        NOTE: Sessions that haven't heartbeated during downtime will be removed.
        If pod was down for >24h, active sessions may be cleaned. This matches
        Tinker SDK behavior where clients must re-establish sessions after restarts.
        """
        with self._lock:
            cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                # Get stale sessions (return IDs for in-memory cleanup)
                cursor.execute(
                    "SELECT session_id FROM sessions WHERE last_heartbeat < ?",
                    (cutoff,)
                )
                stale_ids = [row[0] for row in cursor.fetchall()]

                if not stale_ids:
                    return 0, []

                # Delete samplers for stale sessions
                cursor.execute("""
                    DELETE FROM samplers
                    WHERE session_id IN (SELECT session_id FROM sessions WHERE last_heartbeat < ?)
                """, (cutoff,))

                # Delete session models
                cursor.execute("""
                    DELETE FROM session_models
                    WHERE session_id IN (SELECT session_id FROM sessions WHERE last_heartbeat < ?)
                """, (cutoff,))

                # Delete stale sessions
                cursor.execute(
                    "DELETE FROM sessions WHERE last_heartbeat < ?",
                    (cutoff,)
                )

                conn.commit()
                logger.info(f"Cleaned up {len(stale_ids)} stale sessions from database")
                return len(stale_ids), stale_ids
            finally:
                conn.close()

    def get_stats(self) -> Dict[str, int]:
        """
        Get storage statistics.

        Returns:
            Dict with counts for sessions, samplers, and models
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM sessions")
                session_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM samplers")
                sampler_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM session_models")
                model_count = cursor.fetchone()[0]

                return {
                    "sessions": session_count,
                    "samplers": sampler_count,
                    "session_models": model_count
                }
            finally:
                conn.close()
