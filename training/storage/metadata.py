"""
Metadata storage module for training runs and checkpoints.

This module manages persistent metadata for training runs and checkpoints,
ensuring that training state survives even after training clients are deleted.
"""
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetadataReadError(Exception):
    pass


def _contained(base: Path, *parts: str) -> Path:
    """base/parts, refused unless the normalised path stays under base. The parts are ids and
    names that arrive in requests, so every path built from them goes through here."""
    root = os.path.normpath(base)
    path = os.path.normpath(os.path.join(root, *parts))
    if not path.startswith(root + os.sep):
        raise ValueError(f"{'/'.join(parts)!r} leaves {root}")
    return Path(path)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Replace ``path`` with ``data`` as JSON, or leave it exactly as it was.

    The text goes to a temp file beside the record and is renamed over it, so a
    reader sees the old record or the new one and never a partial one. A crash
    between the two leaves the temp file; ``sweep_temp_files`` removes it at boot.
    The temp name does not include the record's name, so it fits wherever the record does.
    """
    text = json.dumps(data, indent=2, sort_keys=True)
    tmp = path.with_name(f".{uuid.uuid4().hex}.tmp")   # not *.json: no listing picks it up
    try:
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class MetadataStorage:
    """
    Manages metadata for training runs and checkpoints.

    This class provides file-based storage for training run metadata
    and checkpoint management, ensuring persistence across sessions.
    """

    def __init__(
        self,
        metadata_dir: Path,
        training_runs_dir: Optional[Path] = None,
        checkpoints_dir: Optional[Path] = None
    ):
        """
        Initialize metadata storage.

        Args:
            metadata_dir: Base directory for metadata storage
            training_runs_dir: Optional override for training runs directory
            checkpoints_dir: Optional override for checkpoints directory
        """
        self.metadata_dir = Path(metadata_dir)
        self.training_runs_dir = training_runs_dir or self.metadata_dir / "training_runs"
        self.checkpoints_dir = checkpoints_dir or self.metadata_dir / "checkpoints"

        self._init_directories()

    def _init_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.training_runs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized metadata storage at {self.metadata_dir}")

    def sweep_temp_files(self) -> int:
        """Remove the temp files of writes that a crash cut short; returns the count.

        Only for boot, before any writer runs. A leftover ``.<hex>.tmp`` would
        otherwise keep its model directory from being removed, and so keep the
        model listed with no record in it. A model directory this leaves empty is
        removed too.
        """
        removed = 0
        for directory in [self.training_runs_dir, *(d for d in self.checkpoints_dir.glob("*") if d.is_dir())]:
            for tmp in directory.glob(".*.tmp"):
                tmp.unlink()
                removed += 1
            if directory != self.training_runs_dir and not any(directory.iterdir()):
                directory.rmdir()
        if removed:
            logger.warning("Removed %d temp file(s) left by an interrupted metadata write", removed)
        return removed

    def save_training_run(
        self,
        training_run_id: str,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Save metadata for a training run.

        Args:
            training_run_id: Unique identifier for the training run
            metadata: Training run metadata (config, model info, etc.)

        Returns:
            Path to the saved metadata file
        """
        # Ensure required fields
        metadata["training_run_id"] = training_run_id
        metadata["created_at"] = metadata.get(
            "created_at",
            datetime.utcnow().isoformat()
        )
        metadata["updated_at"] = datetime.utcnow().isoformat()

        # Save to JSON file
        metadata_path = _contained(self.training_runs_dir, f"{training_run_id}.json")
        try:
            _write_json(metadata_path, metadata)
            logger.info(f"Saved training run metadata to {metadata_path}")
            return metadata_path
        except Exception as e:
            logger.error(f"Failed to save training run {training_run_id}: {e}")
            raise

    def load_training_run(
        self,
        training_run_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a training run.

        Args:
            training_run_id: Training run identifier

        Returns:
            Training run metadata dict or None if not found
        """
        metadata_path = _contained(self.training_runs_dir, f"{training_run_id}.json")

        if not metadata_path.exists():
            logger.warning(f"Training run {training_run_id} not found")
            return None

        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load training run {training_run_id}: {e}")
            raise MetadataReadError(f"Failed to load training run {training_run_id}") from e

    def update_training_run(
        self,
        training_run_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update existing training run metadata.

        Args:
            training_run_id: Training run identifier
            updates: Fields to update

        Returns:
            True if successful, False if training run not found
        """
        metadata = self.load_training_run(training_run_id)
        if not metadata:
            return False

        # Apply updates
        metadata.update(updates)
        metadata["updated_at"] = datetime.utcnow().isoformat()

        # Save back
        self.save_training_run(training_run_id, metadata)
        return True

    def list_training_runs(
        self,
        model_id: Optional[str] = None,
        base_model: Optional[str] = None,
        *,
        limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        """
        List all training runs with optional filters.

        Args:
            model_id: Filter by model ID
            base_model: Filter by base model name
            limit: Maximum number of results; None returns every run

        Returns:
            List of training run metadata dicts
        """
        runs = []

        # Load all training run files
        for metadata_path in sorted(
            self.training_runs_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Most recent first
        ):
            if limit is not None and len(runs) >= limit:
                break

            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # Apply filters
                if model_id and metadata.get("model_id") != model_id:
                    continue
                if base_model and metadata.get("base_model") != base_model:
                    continue

                runs.append(metadata)
            except Exception as e:
                logger.error(f"Failed to load {metadata_path}: {e}")
                continue

        return runs

    def save_checkpoint(
        self,
        model_id: str,
        checkpoint_name: str,
        checkpoint_data: Dict[str, Any]
    ) -> Path:
        """
        Save checkpoint metadata.

        Args:
            model_id: Model identifier
            checkpoint_name: Checkpoint name
            checkpoint_data: Checkpoint metadata (path, metrics, etc.)

        Returns:
            Path to checkpoint metadata file
        """
        # Create model checkpoint directory
        model_checkpoints_dir = _contained(self.checkpoints_dir, model_id)
        model_checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Add metadata
        checkpoint_data["model_id"] = model_id
        checkpoint_data["checkpoint_name"] = checkpoint_name
        checkpoint_data.setdefault("created_at", datetime.utcnow().isoformat())

        # Save checkpoint metadata
        checkpoint_path = _contained(model_checkpoints_dir, f"{checkpoint_name}.json")
        try:
            _write_json(checkpoint_path, checkpoint_data)
            logger.info(f"Saved checkpoint metadata to {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_name}: {e}")
            raise

    def load_checkpoint(
        self,
        model_id: str,
        checkpoint_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint metadata.

        Args:
            model_id: Model identifier
            checkpoint_name: Checkpoint name

        Returns:
            Checkpoint metadata dict or None if not found
        """
        checkpoint_path = _contained(self.checkpoints_dir, model_id, f"{checkpoint_name}.json")

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint {checkpoint_name} not found for model {model_id}")
            return None

        try:
            with open(checkpoint_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_name}: {e}")
            raise MetadataReadError(f"Failed to load checkpoint {model_id}/{checkpoint_name}") from e

    def list_checkpoints(
        self,
        model_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List all checkpoints for a model.

        Args:
            model_id: Model identifier
            limit: Maximum number of results

        Returns:
            List of checkpoint metadata dicts
        """
        model_checkpoints_dir = _contained(self.checkpoints_dir, model_id)

        if not model_checkpoints_dir.exists():
            return []

        checkpoints = []
        for checkpoint_path in sorted(
            model_checkpoints_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Most recent first
        ):
            if len(checkpoints) >= limit:
                break

            try:
                with open(checkpoint_path, "r") as f:
                    checkpoints.append(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load {checkpoint_path}: {e}")
                continue

        return checkpoints

    def delete_checkpoint(
        self,
        model_id: str,
        checkpoint_name: str
    ) -> bool:
        """
        Delete checkpoint metadata.

        Args:
            model_id: Model identifier
            checkpoint_name: Checkpoint name

        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = _contained(self.checkpoints_dir, model_id, f"{checkpoint_name}.json")

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint {checkpoint_name} not found for deletion")
            return False

        try:
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint {checkpoint_name} for model {model_id}")

            # Clean up empty directory
            model_checkpoints_dir = checkpoint_path.parent
            if model_checkpoints_dir.exists() and not any(model_checkpoints_dir.iterdir()):
                model_checkpoints_dir.rmdir()

            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_name}: {e}")
            raise

    def get_stats(self) -> Dict[str, int]:
        """
        Get storage statistics.

        Returns:
            Dict with counts of training runs and checkpoints
        """
        stats = {
            "training_runs": len(list(self.training_runs_dir.glob("*.json"))),
            "models_with_checkpoints": 0,
            "total_checkpoints": 0
        }

        # Count checkpoints
        if self.checkpoints_dir.exists():
            for model_dir in self.checkpoints_dir.iterdir():
                if model_dir.is_dir():
                    checkpoint_count = len(list(model_dir.glob("*.json")))
                    if checkpoint_count > 0:
                        stats["models_with_checkpoints"] += 1
                        stats["total_checkpoints"] += checkpoint_count

        return stats
