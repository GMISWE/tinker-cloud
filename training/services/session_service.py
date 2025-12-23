"""
Session Service - Basic Session Tracking for Tinker Integration

Tracks:
- Active sessions with metadata
- Models linked to sessions
- Sampling sessions linked to sessions
- Heartbeat timestamps (warn-only on stale)
- Persistent storage via SessionStorage
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage.session_storage import SessionStorage

logger = logging.getLogger(__name__)


@dataclass
class SamplerInfo:
    """Information about a sampling session/sampler."""
    sampler_id: str
    session_id: str
    base_model: Optional[str] = None
    model_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SessionInfo:
    """Information about an active session."""
    session_id: str
    tags: List[str]
    user_metadata: Dict[str, Any]
    sdk_version: str
    created_at: datetime
    last_heartbeat: datetime
    model_ids: List[str] = field(default_factory=list)
    sampling_session_ids: List[str] = field(default_factory=list)
    model_seq_ids: Optional[Dict[str, int]] = field(default=None)  # model_id -> seq_id mapping


class SessionService:
    """
    Session tracking service with optional persistence.

    Provides:
    - Session lifecycle management (create, heartbeat)
    - Model-to-session linking with ordering
    - Sampling session tracking
    - Persistent storage via SessionStorage
    - Stale session cleanup with TTL
    """

    def __init__(
        self,
        storage: Optional["SessionStorage"] = None,
        heartbeat_warn_threshold_sec: int = 120
    ):
        """
        Initialize SessionService.

        Args:
            storage: Optional SessionStorage for persistence
            heartbeat_warn_threshold_sec: Seconds before warning about missed heartbeats
        """
        self._sessions: Dict[str, SessionInfo] = {}
        self._samplers: Dict[str, SamplerInfo] = {}
        self._model_to_session: Dict[str, str] = {}  # model_id -> session_id mapping
        self._storage = storage
        self._heartbeat_warn_threshold = heartbeat_warn_threshold_sec

        # Load persisted data on startup
        if storage:
            self._load_from_storage()

        logger.info(
            f"SessionService initialized (warn threshold: {heartbeat_warn_threshold_sec}s, "
            f"persistence={'enabled' if storage else 'disabled'})"
        )

    def _load_from_storage(self) -> None:
        """Load all sessions and samplers from storage on startup."""
        if not self._storage:
            return

        # Load sessions
        for session_data in self._storage.list_sessions(limit=10000, offset=0):
            session_id = session_data["session_id"]

            # Load models for this session, ORDERED BY model_seq_id
            models = self._storage.get_models_by_session(session_id)
            model_ids = [m["model_id"] for m in models]
            model_seq_ids = {m["model_id"]: m["model_seq_id"] for m in models if m["model_seq_id"] is not None}

            # Build model -> session mapping
            for m in models:
                self._model_to_session[m["model_id"]] = session_id

            # Parse datetime strings
            created_at = datetime.fromisoformat(session_data["created_at"])
            last_heartbeat = datetime.fromisoformat(session_data["last_heartbeat"])

            # Load sampling session IDs for this session
            samplers = self._storage.list_samplers_by_session(session_id)
            sampling_session_ids = [s["sampler_id"] for s in samplers]

            self._sessions[session_id] = SessionInfo(
                session_id=session_id,
                tags=session_data.get("tags", []),
                user_metadata=session_data.get("user_metadata", {}),
                sdk_version=session_data.get("sdk_version", ""),
                created_at=created_at,
                last_heartbeat=last_heartbeat,
                model_ids=model_ids,
                sampling_session_ids=sampling_session_ids,
                model_seq_ids=model_seq_ids if model_seq_ids else None,
            )

            # Load samplers for this session
            for sampler_data in samplers:
                self._samplers[sampler_data["sampler_id"]] = SamplerInfo(
                    sampler_id=sampler_data["sampler_id"],
                    session_id=session_id,
                    base_model=sampler_data.get("base_model"),
                    model_path=sampler_data.get("model_path"),
                )

        logger.info(
            f"Loaded {len(self._sessions)} sessions and {len(self._samplers)} samplers from storage"
        )

    def create_session(
        self,
        session_id: str,
        tags: List[str],
        user_metadata: Dict[str, Any],
        sdk_version: str
    ) -> SessionInfo:
        """
        Create and track a new session.

        Args:
            session_id: Unique session identifier
            tags: Client-provided tags
            user_metadata: Client-provided metadata
            sdk_version: Client SDK version

        Returns:
            SessionInfo for the created session
        """
        now = datetime.now()
        session = SessionInfo(
            session_id=session_id,
            tags=tags,
            user_metadata=user_metadata or {},
            sdk_version=sdk_version,
            created_at=now,
            last_heartbeat=now,
        )
        self._sessions[session_id] = session

        # Persist to storage
        if self._storage:
            self._storage.save_session(
                session_id=session_id,
                sdk_version=sdk_version,
                tags=tags,
                user_metadata=user_metadata or {},
                created_at=now,
                last_heartbeat=now
            )

        logger.info(
            f"Session created: {session_id} "
            f"(sdk={sdk_version}, tags={tags})"
        )
        return session

    def heartbeat(self, session_id: str) -> bool:
        """
        Update heartbeat timestamp for a session.

        Persists to storage for TTL consistency.

        Args:
            session_id: Session to heartbeat

        Returns:
            True if session exists and was updated, False otherwise
        """
        session = self._sessions.get(session_id)
        if session is None:
            logger.warning(f"Heartbeat for unknown session: {session_id}")
            return False

        now = datetime.now()
        time_since_last = (now - session.last_heartbeat).total_seconds()

        if time_since_last > self._heartbeat_warn_threshold:
            logger.warning(
                f"Session {session_id} heartbeat after {time_since_last:.1f}s gap "
                f"(threshold: {self._heartbeat_warn_threshold}s)"
            )

        session.last_heartbeat = now

        # Persist to storage for TTL consistency
        if self._storage:
            self._storage.update_heartbeat(session_id)

        return True

    def add_model(
        self,
        session_id: str,
        model_id: str,
        model_seq_id: int,
        base_model: str,
        model_path: Optional[str] = None
    ) -> None:
        """
        Link a model to a session with ordering and context for matching.

        Inserts in model_seq_id order (not append) to keep list sorted.
        Guards against duplicates.
        Persists base_model/model_path for context matching.

        Args:
            session_id: Session that owns the model
            model_id: Model identifier
            model_seq_id: Sequence ID for ordering within session
            base_model: Base model name (for context matching)
            model_path: Optional checkpoint path (for context matching)

        Raises:
            ValueError: If session not found
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        # Guard against duplicates
        if model_id in session.model_ids:
            logger.warning(f"Model {model_id} already in session {session_id}")
            return

        # Initialize model_seq_ids if needed
        if session.model_seq_ids is None:
            session.model_seq_ids = {}
        session.model_seq_ids[model_id] = model_seq_id

        # Insert at correct position by seq_id to keep model_ids sorted
        insert_idx = 0
        for i, existing_id in enumerate(session.model_ids):
            existing_seq = session.model_seq_ids.get(existing_id, 0)
            if model_seq_id < existing_seq:
                break
            insert_idx = i + 1
        session.model_ids.insert(insert_idx, model_id)

        # Track model -> session mapping
        self._model_to_session[model_id] = session_id

        # Persist to storage
        if self._storage:
            self._storage.add_model_to_session(
                session_id=session_id,
                model_id=model_id,
                model_seq_id=model_seq_id,
                base_model=base_model,
                model_path=model_path
            )

        logger.info(f"Model {model_id} (seq={model_seq_id}) linked to session {session_id}")

    def add_sampling_session(
        self,
        session_id: str,
        sampling_session_id: str,
        base_model: Optional[str] = None,
        model_path: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> bool:
        """
        Link a sampling session to a parent session.

        Uses None for unknown values, not empty string.
        Persists to storage when base_model or model_path provided.

        Args:
            session_id: Parent session
            sampling_session_id: Sampling session identifier
            base_model: Base model name for the sampler (None if unknown)
            model_path: Optional model path (tinker:// URI or checkpoint path)
            model_id: Optional model that created this sampler

        Returns:
            True if session exists and sampler was added, False otherwise
        """
        session = self._sessions.get(session_id)
        if session is None:
            logger.warning(
                f"Adding sampling session {sampling_session_id} "
                f"to unknown session {session_id}"
            )
            return False

        session.sampling_session_ids.append(sampling_session_id)

        # Store sampler metadata (always store if base_model or model_path provided)
        # Use None for unknown values, not empty string
        if base_model or model_path:
            self._samplers[sampling_session_id] = SamplerInfo(
                sampler_id=sampling_session_id,
                session_id=session_id,
                base_model=base_model if base_model else None,
                model_path=model_path,
            )

            # Persist to storage
            if self._storage:
                self._storage.save_sampler(
                    sampler_id=sampling_session_id,
                    session_id=session_id,
                    model_id=model_id,
                    base_model=base_model if base_model else None,
                    model_path=model_path
                )

        logger.debug(
            f"Sampling session {sampling_session_id} "
            f"linked to session {session_id} (base_model={base_model}, model_path={model_path})"
        )
        return True

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get session info by ID.

        Args:
            session_id: Session to look up

        Returns:
            SessionInfo if found, None otherwise
        """
        return self._sessions.get(session_id)

    def get_stale_sessions(self) -> List[str]:
        """
        Get list of sessions that have missed heartbeats.

        Returns:
            List of session_ids that haven't had a heartbeat
            within the warning threshold
        """
        now = datetime.now()
        stale = []
        for session_id, session in self._sessions.items():
            age = (now - session.last_heartbeat).total_seconds()
            if age > self._heartbeat_warn_threshold:
                stale.append(session_id)
        return stale

    def get_active_session_count(self) -> int:
        """Get count of tracked sessions."""
        return len(self._sessions)

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists (in-memory or storage)."""
        if session_id in self._sessions:
            return True
        # Fallback to storage check
        if self._storage and self._storage.session_exists(session_id):
            return True
        return False

    def list_sessions(self, limit: int = 20, offset: int = 0) -> List[str]:
        """
        List session IDs with pagination.

        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            List of session IDs
        """
        session_ids = list(self._sessions.keys())
        return session_ids[offset:offset + limit]

    def get_sampler(self, sampler_id: str) -> Optional[SamplerInfo]:
        """
        Get sampler info by ID.

        Args:
            sampler_id: Sampler (sampling_session_id) to look up

        Returns:
            SamplerInfo if found, None otherwise
        """
        return self._samplers.get(sampler_id)

    def register_ephemeral_sampler(
        self,
        sampler_id: str,
        model_id: str,
        base_model: Optional[str] = None,
        model_path: Optional[str] = None
    ) -> bool:
        """
        Register an ephemeral sampler from save_weights_for_sampler.

        Gets session from persisted model->session mapping.
        Uses None instead of empty string for unknown base_model.
        Persists to storage.

        Args:
            sampler_id: The sampling_session_id to register
            model_id: The model that created this sampler
            base_model: Base model name (None if unknown)
            model_path: Optional checkpoint path

        Returns:
            True if session found and sampler registered, False otherwise
        """
        # Find session that owns this model (use persisted mapping)
        session_id = self._model_to_session.get(model_id)
        if not session_id:
            logger.warning(f"Cannot register sampler {sampler_id}: model {model_id} not in any session")
            return False

        session = self._sessions.get(session_id)
        if session:
            session.sampling_session_ids.append(sampler_id)

        # Use None for unknown values, not empty string
        self._samplers[sampler_id] = SamplerInfo(
            sampler_id=sampler_id,
            session_id=session_id,
            base_model=base_model if base_model else None,
            model_path=model_path,
        )

        # Persist to storage
        if self._storage:
            self._storage.save_sampler(
                sampler_id=sampler_id,
                session_id=session_id,
                model_id=model_id,
                base_model=base_model if base_model else None,
                model_path=model_path
            )

        logger.info(
            f"Ephemeral sampler {sampler_id} registered to session {session_id} "
            f"(from model {model_id})"
        )
        return True

    def get_session_for_model(self, model_id: str) -> Optional[str]:
        """
        Get session_id that owns this model.

        Uses persisted model->session mapping.

        Args:
            model_id: Model identifier

        Returns:
            Session ID or None if not found
        """
        return self._model_to_session.get(model_id)

    def remove_model(self, model_id: str) -> Optional[str]:
        """
        Remove a model from its session.

        Discovers session via storage lookup if not in memory.

        Args:
            model_id: The model ID to remove

        Returns:
            The session_id that owned the model, or None if not found
        """
        # Find session_id from in-memory mapping first
        session_id = self._model_to_session.pop(model_id, None)

        # Remove from storage (returns session_id if found)
        if self._storage:
            stored_session_id = self._storage.remove_model_from_session(model_id)
            if session_id is None:
                session_id = stored_session_id

        if session_id is None:
            logger.debug(f"Model {model_id} not found in any session")
            return None

        # Remove from session's model_ids
        session = self._sessions.get(session_id)
        if session and model_id in session.model_ids:
            session.model_ids.remove(model_id)
            if session.model_seq_ids:
                session.model_seq_ids.pop(model_id, None)

        logger.info(f"Model {model_id} removed from session {session_id}")
        return session_id

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary dict for a session (useful for debugging/logging).

        Args:
            session_id: Session to summarize

        Returns:
            Dict with session summary or None if not found
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        now = datetime.now()
        return {
            "session_id": session.session_id,
            "sdk_version": session.sdk_version,
            "tags": session.tags,
            "created_at": session.created_at.isoformat(),
            "last_heartbeat": session.last_heartbeat.isoformat(),
            "heartbeat_age_sec": (now - session.last_heartbeat).total_seconds(),
            "model_count": len(session.model_ids),
            "sampling_session_count": len(session.sampling_session_ids),
        }

    def cleanup_stale_sessions(self, max_age_hours: int = 24) -> Tuple[int, List[str]]:
        """
        Cleanup stale sessions from storage AND in-memory cache.

        Args:
            max_age_hours: Maximum age in hours before session is considered stale

        Returns:
            Tuple of (count_removed, list_of_removed_session_ids)
        """
        if not self._storage:
            return 0, []

        count, stale_ids = self._storage.cleanup_stale_sessions(max_age_hours)

        # Log warning about potential orphans before removal
        for session_id in stale_ids:
            session = self._sessions.get(session_id)
            if session and session.model_ids:
                logger.warning(
                    f"Session {session_id} cleaned with {len(session.model_ids)} models still registered. "
                    f"Model IDs: {session.model_ids}. These may be orphaned training_clients."
                )

        # Also drop from in-memory cache
        for session_id in stale_ids:
            if session_id in self._sessions:
                session = self._sessions.pop(session_id)
                # Remove model->session mappings
                for model_id in session.model_ids:
                    self._model_to_session.pop(model_id, None)
                # Remove samplers
                for sampler_id in session.sampling_session_ids:
                    self._samplers.pop(sampler_id, None)

        if count:
            logger.info(f"Cleaned up {count} stale sessions (TTL={max_age_hours}h)")

        return count, stale_ids
