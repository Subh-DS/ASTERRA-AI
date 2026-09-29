
"""FastAPI application for the ASTERRA production pipeline."""

from contextlib import asynccontextmanager
import asyncio
import json
from pathlib import Path
import sys

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

try:
    from .config import settings
    from .jobs import manager
    from .schemas import HealthResponse, MapJobRequest
    from .storage import paths_for
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.config import settings
    from backend.jobs import manager
    from backend.schemas import HealthResponse, MapJobRequest
    from backend.storage import paths_for


@asynccontextmanager
async def lifespan(_app):
    manager.startup()
    if settings.preload_model:
        try:
            from .services.inference_service import inference_service
        except ImportError:
            from backend.services.inference_service import inference_service
        inference_service._ensure_model(settings)
        try:
            from .segmentation.engine import ENGINE as segmentation_engine
        except ImportError:
            from backend.segmentation.engine import ENGINE as segmentation_engine
        segmentation_engine._load()
    yield
    manager.shutdown()


app = FastAPI(title="ASTERRA AI API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _save_upload(upload, destination, max_bytes, extensions=None):
    destination = Path(destination)
    if extensions and Path(upload.filename or "").suffix.lower() not in extensions:
        raise HTTPException(400, f"Unsupported file type for {upload.filename or 'upload'}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with destination.open("wb") as target:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                destination.unlink(missing_ok=True)
                raise HTTPException(413, "Upload exceeds the configured size limit")
            target.write(chunk)
    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, "Upload is empty")
    return total


def _parse_gcps(raw):
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "gcps must be valid JSON") from exc
    if not isinstance(value, list):
        raise HTTPException(400, "gcps must be a JSON array")
    return value


def _job(job_id):
    try:
        return manager.store.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, "unknown job") from exc


def _authorize(job, token):
    if token and token != job.get("file_token"):
        raise HTTPException(403, "invalid file token")
    if settings.require_file_token and token != job.get("file_token"):
        raise HTTPException(403, "file token required")


def _artifact(job_id, name):
    paths = manager.store.paths(job_id)
    result = manager.store.get(job_id).get("result") or {}
    mapping = {
        "dsm.tif": paths.metric_dsm if result.get("metric") else paths.relative_dsm,
        "model.glb": paths.model_glb,
        "buildings.geojson": paths.buildings_geojson,
        "environment.geojson": paths.environment_geojson,
        "mask.png": paths.semantic_preview,
        "metadata.json": paths.root / "metadata.json",
        "calibration-metadata.json": paths.calibration_metadata,
    }
    if name not in mapping:
        raise HTTPException(404, "unknown artifact")
    path = mapping[name]
    if not path.exists():
        raise HTTPException(404, "artifact is not ready")
    return path


@app.get("/api/health", response_model=HealthResponse)
def health():
    try:
        from .services.inference_service import inference_service
    except ImportError:
        from backend.services.inference_service import inference_service
    engine = inference_service.health(settings)
    try:
        from .segmentation.engine import ENGINE as segmentation_engine
    except ImportError:
        from backend.segmentation.engine import ENGINE as segmentation_engine
    engine = {**engine, "segmentation": segmentation_engine.status()}
    return {"status": "ok" if engine.get("checkpoint_exists") and engine.get("depth_backend_available") else "degraded", "engine": engine, "jobs_queued": manager.queue_depth(), "max_jobs": settings.max_queue_size}


@app.get("/api/engine")
def engine():
    return health()


def _imagery_catalog():
    try:
        from .services.imagery_service import imagery_catalog
    except ImportError:
        from backend.services.imagery_service import imagery_catalog
    return imagery_catalog(settings)


@app.get("/api/imagery/providers")
def imagery_providers():
    return {"providers": _imagery_catalog().descriptors()}


@app.get("/api/imagery/search")
def imagery_search(
    north: float,
    south: float,
    east: float,
    west: float,
    provider: str = "auto",
    max_cloud: float | None = None,
):
    try:
        return _imagery_catalog().search(
            {"north": north, "south": south, "east": east, "west": west},
            provider,
            max_cloud,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        try:
            from .services.imagery_service import ImageryValidationError
        except ImportError:
            from backend.services.imagery_service import ImageryValidationError
        if isinstance(exc, ImageryValidationError):
            raise HTTPException(400, str(exc)) from exc
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/map-jobs")
def create_map_job_endpoint(request: MapJobRequest):
    try:
        try:
            from .services.imagery_service import aoi_payload, imagery_catalog
        except ImportError:
            from backend.services.imagery_service import aoi_payload, imagery_catalog
        aoi = aoi_payload(request.aoi.model_dump())
        catalog = imagery_catalog(settings)
        # Validate the provider before creating a persistent job.  The scene
        # itself is resolved and downloaded in the worker so this endpoint
        # remains responsive while large COG assets are materialized.
        catalog.get(request.provider)
    except Exception as exc:
        try:
            from .services.imagery_service import ImageryValidationError
        except ImportError:
            from backend.services.imagery_service import ImageryValidationError
        if isinstance(exc, (ValueError, ImageryValidationError)):
            raise HTTPException(400, str(exc)) from exc
        raise HTTPException(502, str(exc)) from exc

    job = manager.create(
        "map-scene.tif",
        "image/tiff",
        [],
        settings.dem_provider,
    )
    manager.store.update(
        job["job_id"],
        source_type="map",
        aoi=aoi,
        imagery_provider=request.provider,
        imagery_item_id=request.item_id,
        imagery_max_cloud=request.max_cloud,
        imagery_quality=request.quality,
    )
    try:
        manager.start(job["job_id"])
    except RuntimeError as exc:
        manager.store.fail(job["job_id"], str(exc), "ingest")
        raise HTTPException(429, str(exc)) from exc
    return {"job_id": job["job_id"], "status": "queued", "file_token": job["file_token"], "reused": False}


@app.post("/api/jobs")
async def create_job_endpoint(
    image: UploadFile = File(...),
    gcps: str = Form(default="[]"),
    dem_provider: str = Form(default="auto"),
    dem: UploadFile | None = File(default=None),
):
    parsed_gcps = _parse_gcps(gcps)
    job = manager.create(image.filename or "upload", image.content_type, parsed_gcps, dem_provider)
    paths = manager.store.paths(job["job_id"])
    try:
        await _save_upload(image, paths.source_upload, settings.max_upload_bytes, {".tif", ".tiff", ".png", ".jpg", ".jpeg"})
        if dem is not None:
            await _save_upload(dem, paths.dem_upload, settings.max_upload_bytes, {".tif", ".tiff"})
    except Exception as exc:
        manager.store.fail(job["job_id"], str(exc), "ingest")
        raise
    try:
        manager.start(job["job_id"])
    except RuntimeError as exc:
        manager.store.fail(job["job_id"], str(exc), "ingest")
        raise HTTPException(429, str(exc)) from exc
    return {"job_id": job["job_id"], "status": "queued", "file_token": job["file_token"]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job(job_id)


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    _job(job_id)
    try:
        manager.retry(job_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(429, str(exc)) from exc
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}/dsm")
def dsm(job_id: str, token: str | None = None):
    job = _job(job_id); _authorize(job, token)
    return FileResponse(_artifact(job_id, "dsm.tif"), media_type="image/tiff", filename="dsm.tif")


@app.get("/api/jobs/{job_id}/dsm.bin")
def dsm_binary(job_id: str, token: str | None = None):
    job = _job(job_id); _authorize(job, token)
    import rasterio
    import numpy as np
    with rasterio.open(_artifact(job_id, "dsm.tif")) as src:
        data = src.read(1).astype(np.float32)
    return Response(data.tobytes(order="C"), media_type="application/octet-stream", headers={"X-DSM-Width": str(data.shape[1]), "X-DSM-Height": str(data.shape[0])})


@app.get("/api/jobs/{job_id}/export/{artifact_name:path}")
def export_artifact(job_id: str, artifact_name: str, token: str | None = None):
    job = _job(job_id); _authorize(job, token)
    path = _artifact(job_id, artifact_name)
    media = {"dsm.tif": "image/tiff", "model.glb": "model/gltf-binary", "buildings.geojson": "application/geo+json", "environment.geojson": "application/geo+json", "mask.png": "image/png", "metadata.json": "application/json", "calibration-metadata.json": "application/json"}.get(artifact_name, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=Path(artifact_name).name)


@app.get("/files/{job_id}/{file_name}")
def preview_file(job_id: str, file_name: str, token: str | None = None):
    job = _job(job_id); _authorize(job, token)
    paths = manager.store.paths(job_id)
    mapping = {"texture.jpg": paths.texture, "normal.png": paths.normal}
    if file_name not in mapping or not mapping[file_name].exists():
        raise HTTPException(404, "preview is not ready")
    return FileResponse(mapping[file_name], media_type="image/jpeg" if file_name.endswith(".jpg") else "image/png")


async def _ws_job(websocket: WebSocket, job_id: str):
    try:
        current = manager.store.get(job_id)
        await websocket.accept()
        queue, subscription, history = await manager.store.subscribe(job_id)
        for event in history:
            await websocket.send_json(event)
        if current["status"] == "complete":
            await websocket.send_json({"type": "job_complete", "job_id": job_id, "status": "complete", "result": current.get("result")})
        elif current["status"] == "error":
            await websocket.send_json({"type": "job_error", "job_id": job_id, "status": "error", "message": current.get("message")})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
                await websocket.send_json(event)
                if event.get("type") in {"job_complete", "job_error"}:
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "job_id": job_id})
    except KeyError:
        await websocket.close(code=4404)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            manager.store.unsubscribe(job_id, subscription)
        except (UnboundLocalError, KeyError):
            pass


@app.websocket("/api/jobs/{job_id}/ws")
async def job_socket(websocket: WebSocket, job_id: str):
    await _ws_job(websocket, job_id)


@app.websocket("/api/ws/jobs/{job_id}")
async def frontend_job_socket(websocket: WebSocket, job_id: str):
    await _ws_job(websocket, job_id)


@app.post("/api/chat")
async def chat_endpoint(payload: dict):
    """Geospatial-only chat endpoint for the field assistant."""
    try:
        from .services.chat_service import chat_service
    except ImportError:
        from backend.services.chat_service import chat_service

    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        raise HTTPException(400, "messages must be a list")

    # Validate message format
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise HTTPException(400, "each message must have 'role' and 'content'")
        if msg["role"] not in ("user", "assistant"):
            raise HTTPException(400, "role must be 'user' or 'assistant'")

    reply = chat_service.reply(messages)
    return {"reply": reply}


@app.post("/api/jobs/{job_id}/validate")
async def validate_reference(job_id: str, reference: UploadFile = File(...)):
    job = _job(job_id)
    paths = manager.store.paths(job_id)
    target = paths.input_dir / "validation_reference.tif"
    await _save_upload(reference, target, settings.max_upload_bytes, {".tif", ".tiff"})
    try:
        import rasterio
        import numpy as np
        with rasterio.open(target) as src:
            if src.count < 1 or src.width * src.height > settings.max_raster_pixels:
                raise ValueError("reference raster is invalid or too large")
            values = src.read(1, masked=True)
            if not np.isfinite(values.compressed()).all():
                raise ValueError("reference raster has no finite values")
        return {"valid": True, "job_id": job_id, "reference": {"width": src.width, "height": src.height, "crs": None if src.crs is None else str(src.crs)}}
    except Exception as exc:
        raise HTTPException(400, f"invalid reference raster: {exc}") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=False)
