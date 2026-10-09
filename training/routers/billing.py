"""
Billing Router - Usage Queries

Endpoints:
- GET /api/v1/billing/usage/checkpoints/current - Current checkpoint storage usage
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends

from ..checkpoints import CheckpointStore
from ..core.dependencies import get_checkpoint_store, verify_api_key_dep
from ..models.responses import (
    CheckpointStorageUsageItem,
    CurrentCheckpointStorageUsageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["billing"])

GIGABYTE = 2 ** 30


@router.get("/api/v1/billing/usage/checkpoints/current", response_model=CurrentCheckpointStorageUsageResponse)
async def current_checkpoint_storage_usage(
    project_id: Optional[str] = None,
    _: None = Depends(verify_api_key_dep),
    store: CheckpointStore = Depends(get_checkpoint_store),
):
    """One row aggregating every completed persistent checkpoint on this
    server. Projects and owners are not tracked, so project_id does not
    filter; it is echoed. No storage rate is configured: rate and cost are null."""
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
