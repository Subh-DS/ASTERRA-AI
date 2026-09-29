"""Environment-backed service configuration."""

from dataclasses import dataclass, field
import os
from pathlib import Path


def _bool(name, default=False):
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name, default):
    value = os.getenv(name)
    return tuple(item.strip() for item in value.split(",") if item.strip()) if value else tuple(default)


def apply_torch_threads():
    """Keep optional segmentation inference from oversubscribing the worker."""
    try:
        import torch
        if not torch.cuda.is_available():
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    except Exception:
        pass


@dataclass(frozen=True)
class Settings:
    project_root: Path
    backend_root: Path
    jobs_root: Path
    runtime_root: Path
    model_path: Path
    depth_anything_root: Path
    allowed_origins: tuple[str, ...] = field(default_factory=tuple)
    max_upload_bytes: int = 1024 * 1024 * 1024
    max_raster_pixels: int = 100_000_000
    min_raster_dimension: int = 128
    max_queue_size: int = 8
    worker_count: int = 1
    mesh_size: int = 512
    dem_provider: str = "auto"
    dem_path: Path | None = None
    dem_online: bool = True
    dem_cache_root: Path | None = None
    dem_download_timeout: float = 30.0
    dem_max_download_bytes: int = 64 * 1024 * 1024
    imagery_stac_url: str = "https://earth-search.aws.element84.com/v1"
    imagery_collections: tuple[str, ...] = ("sentinel-2-c1-l2a", "sentinel-2-l2a")
    imagery_default_provider: str = "public"
    imagery_esri_url: str | None = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export,https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    imagery_highres_stac_url: str | None = None
    imagery_highres_collection: str | None = None
    imagery_highres_token: str | None = None
    imagery_cache_root: Path | None = None
    imagery_timeout: float = 30.0
    imagery_max_candidates: int = 6
    buildings_enabled: bool = True
    buildings_overpass_url: str | None = "https://overpass-api.de/api/interpreter,https://overpass.kumi.systems/api/interpreter"
    buildings_timeout: float = 20.0
    buildings_max: int = 500
    buildings_min_area_m2: float = 4.0
    environment_enabled: bool = True
    environment_max_features: int = 1200
    environment_max_trees: int = 500
    segmentation_enabled: bool = True
    segmentation_model: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    segmentation_confidence: float = 0.55
    segmentation_device: str = "auto"
    segmentation_max_side: int = 1024
    segmentation_timeout_seconds: float = 12.0
    # Segmentation is an enhancement, not a reason to hold the reconstruction
    # queue hostage to a first-run Hugging Face download.  A local-only
    # default fails fast into the deterministic RGB fallback; operators can
    # opt into remote model downloads explicitly.
    segmentation_local_only: bool = True
    segmentation_cache_dir: Path | None = None
    map_mock: bool = False
    preload_model: bool = False
    require_file_token: bool = False
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls):
        project_root = Path(os.getenv("ASTERRA_ROOT", Path(__file__).resolve().parents[1])).resolve()
        backend_root = project_root / "backend"
        model_path = Path(os.getenv(
            "ASTERRA_MODEL_PATH",
            project_root / "models" / "asterra_stage5" / "urban3d_v3_1" / "ASTERRA_FINAL_HEIGHT_MODEL.pth",
        )).resolve()
        dem_value = os.getenv("ASTERRA_DEM_PATH")
        dem_cache_value = os.getenv("ASTERRA_DEM_CACHE")
        imagery_cache_value = os.getenv("ASTERRA_IMAGERY_CACHE")
        segmentation_cache_value = os.getenv("ASTERRA_SEGMENTATION_CACHE")
        return cls(
            project_root=project_root,
            backend_root=backend_root,
            jobs_root=Path(os.getenv("ASTERRA_JOBS_ROOT", project_root / "outputs" / "jobs")).resolve(),
            runtime_root=Path(os.getenv("ASTERRA_RUNTIME_ROOT", backend_root / "runtime")).resolve(),
            model_path=model_path,
            depth_anything_root=Path(os.getenv("ASTERRA_DEPTH_ANYTHING_ROOT", project_root / "external" / "Depth-Anything-V2")).resolve(),
            allowed_origins=_csv("ASTERRA_ALLOWED_ORIGINS", ("http://127.0.0.1:5173", "http://localhost:5173")),
            max_upload_bytes=int(os.getenv("ASTERRA_MAX_UPLOAD_BYTES", 1024 * 1024 * 1024)),
            max_raster_pixels=int(os.getenv("ASTERRA_MAX_RASTER_PIXELS", 100_000_000)),
            min_raster_dimension=max(0, int(os.getenv("ASTERRA_MIN_RASTER_DIMENSION", 128))),
            max_queue_size=int(os.getenv("ASTERRA_MAX_QUEUE_SIZE", 8)),
            worker_count=max(1, int(os.getenv("ASTERRA_WORKERS", 1))),
            mesh_size=max(32, int(os.getenv("ASTERRA_MESH_SIZE", 512))),
            dem_provider=os.getenv("ASTERRA_DEM_PROVIDER", "auto").lower(),
            dem_path=Path(dem_value).resolve() if dem_value else None,
            dem_online=_bool("ASTERRA_DEM_ONLINE", True),
            dem_cache_root=Path(dem_cache_value).resolve() if dem_cache_value else (backend_root / "runtime" / "dem_cache"),
            dem_download_timeout=float(os.getenv("ASTERRA_DEM_DOWNLOAD_TIMEOUT", "30")),
            dem_max_download_bytes=int(os.getenv("ASTERRA_DEM_MAX_DOWNLOAD_BYTES", str(64 * 1024 * 1024))),
            imagery_stac_url=os.getenv("ASTERRA_IMAGERY_STAC_URL", "https://earth-search.aws.element84.com/v1"),
            imagery_collections=_csv("ASTERRA_IMAGERY_COLLECTIONS", ("sentinel-2-c1-l2a", "sentinel-2-l2a")),
            imagery_default_provider=os.getenv("ASTERRA_IMAGERY_PROVIDER", "public").lower(),
            imagery_esri_url=os.getenv(
                "ASTERRA_IMAGERY_ESRI_URL",
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export,https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export",
            ) or None,
            imagery_highres_stac_url=os.getenv("ASTERRA_IMAGERY_HIGHRES_STAC_URL") or None,
            imagery_highres_collection=os.getenv("ASTERRA_IMAGERY_HIGHRES_COLLECTION") or None,
            imagery_highres_token=os.getenv("ASTERRA_IMAGERY_HIGHRES_TOKEN") or None,
            imagery_cache_root=Path(imagery_cache_value).resolve() if imagery_cache_value else (backend_root / "runtime" / "imagery_cache"),
            imagery_timeout=float(os.getenv("ASTERRA_IMAGERY_TIMEOUT", "30")),
            imagery_max_candidates=max(1, int(os.getenv("ASTERRA_IMAGERY_MAX_CANDIDATES", "6"))),
            buildings_enabled=_bool("ASTERRA_BUILDINGS_ENABLED", True),
            buildings_overpass_url=os.getenv("ASTERRA_BUILDINGS_OVERPASS_URL", "https://overpass-api.de/api/interpreter,https://overpass.kumi.systems/api/interpreter") or None,
            buildings_timeout=float(os.getenv("ASTERRA_BUILDINGS_TIMEOUT", "20")),
            buildings_max=max(1, int(os.getenv("ASTERRA_BUILDINGS_MAX", "500"))),
            buildings_min_area_m2=max(1.0, float(os.getenv("ASTERRA_BUILDINGS_MIN_AREA_M2", "4"))),
            environment_enabled=_bool("ASTERRA_ENVIRONMENT_ENABLED", True),
            environment_max_features=max(1, int(os.getenv("ASTERRA_ENVIRONMENT_MAX_FEATURES", "1200"))),
            environment_max_trees=max(1, int(os.getenv("ASTERRA_ENVIRONMENT_MAX_TREES", "500"))),
            segmentation_enabled=_bool("ASTERRA_SEGMENTATION_ENABLED", True),
            segmentation_model=os.getenv(
                "ASTERRA_SEGMENTATION_MODEL",
                "nvidia/segformer-b0-finetuned-ade-512-512",
            ),
            segmentation_confidence=min(1.0, max(0.0, float(os.getenv("ASTERRA_SEGMENTATION_CONFIDENCE", "0.55")))),
            segmentation_device=os.getenv("ASTERRA_SEGMENTATION_DEVICE", "auto").lower(),
            segmentation_max_side=max(256, int(os.getenv("ASTERRA_SEGMENTATION_MAX_SIDE", "1024"))),
            segmentation_timeout_seconds=max(1.0, float(os.getenv("ASTERRA_SEGMENTATION_TIMEOUT", "12"))),
            segmentation_local_only=_bool("ASTERRA_SEGMENTATION_LOCAL_ONLY", True),
            segmentation_cache_dir=Path(segmentation_cache_value).resolve() if segmentation_cache_value else (backend_root / "runtime" / "segmentation_cache"),
            map_mock=_bool("DW_MAP_MOCK", False),
            preload_model=_bool("ASTERRA_PRELOAD_MODEL", False),
            require_file_token=_bool("ASTERRA_REQUIRE_FILE_TOKEN", False),
            host=os.getenv("ASTERRA_HOST", "127.0.0.1"),
            port=int(os.getenv("ASTERRA_PORT", 8000)),
        )


settings = Settings.from_env()
