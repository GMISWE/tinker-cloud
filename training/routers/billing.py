"""
Billing Router - Usage Queries

Endpoints:
- GET /api/v1/billing/usage/checkpoints/current - Current checkpoint storage usage
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends

from ..checkpoints import CheckpointStore
from ..core.dependencies import get_checkpoint_store, known_project, verify_api_key_dep
from ..models.responses import (
    CheckpointStorageUsageItem,
    CurrentCheckpointStorageUsageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["billing"])

GIGABYTE = 2 ** 30


@router.get("/api/v1/billing/usage/checkpoints/current", response_model=CurrentCheckpointStorageUsageResponse)
async def current_checkpoint_storage_usage(
    project_id: Optional[str] = Depends(known_project),
    _: None = Depends(verify_api_key_dep),
    store: CheckpointStore = Depends(get_checkpoint_store),
):
    """One row aggregating every completed persistent checkpoint on this server. The server
    has one project (another project_id is a 404, as for training runs), so the row is that
    project's; project_id is echoed. No storage rate is configured: rate and cost are null."""
    count, size_bytes = store.usage()
    return CurrentCheckpointStorageUsageResponse(
        effective_rate_usd_per_gigabyte_month=None,
        data=[CheckpointStorageUsageItem(
            project_id=project_id,
            checkpoint_count=count,
            size_bytes=size_bytes,
            size_gigabytes=size_bytes / GIGABYTE,
            estimated_monthly_cost_usd=None,
        )],
    )
