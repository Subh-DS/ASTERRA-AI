"""Typed API contracts shared by routes and job persistence."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GCPInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    lat: float | None = None
    lon: float | None = None
    elev: float | None = None
    elevation: float | None = None
    x: float | None = None
    y: float | None = None
    pixel_x: float | None = None
    pixel_y: float | None = None
    col: float | None = None
    row: float | None = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "complete", "error"]
    file_token: str


class StageStatus(BaseModel):
    state: Literal["queued", "active", "done", "error"] = "queued"
    progress: int = Field(default=0, ge=0, le=100)
    sub: str = ""


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "complete", "error"]
    stage: str | None = None
    progress: int = Field(default=0, ge=0, le=100)
    stages: dict[str, StageStatus]
    result: dict[str, Any] | None = None
    message: str | None = None
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    engine: dict[str, Any]
    jobs_queued: int
    max_jobs: int


class AOIInput(BaseModel):
    north: float
    south: float
    east: float
    west: float


class MapJobRequest(BaseModel):
    aoi: AOIInput
    provider: str = "auto"
    item_id: str | None = None
    max_cloud: float | None = Field(default=None, ge=0, le=100)
    quality: Literal["low", "medium", "high"] = "medium"

