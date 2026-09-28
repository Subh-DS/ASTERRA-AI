"""Isolated job storage with atomic JSON state."""

from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import re
import secrets
import threading
from uuid import uuid4

JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")
STAGES = ("ingest", "depth", "calibrate", "mesh")

def utc_now():
    return datetime.now(timezone.utc).isoformat()

@dataclass(frozen=True)
class JobPaths:
    root: Path
    input_dir: Path
    inference_dir: Path
    calibration_dir: Path
    reconstruction_dir: Path

    @property
    def job_json(self): return self.root / "job.json"
    @property
    def events_jsonl(self): return self.root / "events.jsonl"
    @property
    def source_upload(self): return self.input_dir / "source.upload"
    @property
    def source_raster(self): return self.input_dir / "source.tif"
    @property
    def dem_upload(self): return self.input_dir / "dem.tif"
    @property
    def gcp_csv(self): return self.input_dir / "gcps.csv"
    @property
    def prediction_npy(self): return self.inference_dir / "prediction.npy"
    @property
    def prediction_tif(self): return self.inference_dir / "prediction.tif"
    @property
    def metric_dsm(self): return self.calibration_dir / "metric_dsm.tif"
    @property
    def relative_dsm(self): return self.calibration_dir / "relative_dsm.tif"
    @property
    def calibration_metadata(self): return self.calibration_dir / "metadata.json"
    @property
    def model_glb(self): return self.reconstruction_dir / "model.glb"
    @property
    def reconstruction_metadata(self): return self.reconstruction_dir / "metadata.json"
    @property
    def buildings_geojson(self): return self.reconstruction_dir / "buildings.geojson"
    @property
    def environment_geojson(self): return self.reconstruction_dir / "environment.geojson"
    @property
    def semantic_mask(self): return self.reconstruction_dir / "mask.tif"
    @property
    def semantic_preview(self): return self.reconstruction_dir / "mask.png"
    @property
    def texture(self): return self.reconstruction_dir / "texture.jpg"
    @property
    def normal(self): return self.reconstruction_dir / "normal.png"

def paths_for(root, job_id):
    if not JOB_ID_RE.fullmatch(job_id): raise ValueError("Invalid job id")
    root = Path(root).resolve() / job_id
    return JobPaths(root, root / "input", root / "inference", root / "calibration", root / "reconstruction")

class JobStore:
    def __init__(self, jobs_root):
        self.jobs_root = Path(jobs_root).resolve()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._events = {}
        self._subscribers = {}

    def _read(self, job_id):
        paths = paths_for(self.jobs_root, job_id)
        if not paths.job_json.exists():
            raise KeyError(job_id)
        return json.loads(paths.job_json.read_text(encoding="utf-8"))

    def _write(self, job):
        paths = paths_for(self.jobs_root, job["job_id"])
        temporary = paths.job_json.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(job, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        temporary.replace(paths.job_json)

    def create(self, original_name, content_type=None, gcps=None, dem_provider="auto", dem_path=None):
        with self._lock:
            job_id = uuid4().hex
            paths = paths_for(self.jobs_root, job_id)
            for directory in (paths.root, paths.input_dir, paths.inference_dir, paths.calibration_dir, paths.reconstruction_dir):
                directory.mkdir(parents=True, exist_ok=True)
            now = utc_now()
            job = {
                "job_id": job_id, "status": "queued", "stage": "ingest", "progress": 0,
                "stages": {stage: {"state": "queued", "progress": 0, "sub": ""} for stage in STAGES},
                "result": None, "message": None, "created_at": now, "updated_at": now,
                "file_token": secrets.token_urlsafe(32), "source_name": Path(original_name or "upload").name,
                "content_type": content_type, "gcps": gcps or [], "dem_provider": dem_provider,
                "dem_path": str(dem_path) if dem_path else None,
            }
            self._write(job)
            self._events[job_id] = []
            self._subscribers[job_id] = set()
            self._emit_locked(job_id, {"type": "job_created", "job_id": job_id, "status": "queued"})
            return copy.deepcopy(job)

    def get(self, job_id):
        with self._lock:
            return copy.deepcopy(self._read(job_id))

    def paths(self, job_id):
        with self._lock:
            self._read(job_id)
            return paths_for(self.jobs_root, job_id)

    def update(self, job_id, **changes):
        with self._lock:
            job = self._read(job_id)
            job.update(changes)
            job["updated_at"] = utc_now()
            self._write(job)
            return copy.deepcopy(job)

    def update_stage(self, job_id, stage, state, progress, sub=""):
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        with self._lock:
            job = self._read(job_id)
            value = max(0, min(100, int(progress)))
            job["stages"][stage] = {"state": state, "progress": value, "sub": sub or ""}
            job["stage"], job["progress"] = stage, value
            if state == "active": job["status"] = "running"
            if state == "error": job["status"], job["message"] = "error", sub or "Pipeline stage failed"
            job["updated_at"] = utc_now()
            self._write(job)
            self._emit_locked(job_id, {"type": "stage", "job_id": job_id, "stage": stage, "status": state, "progress": value, "sub": sub or ""})

    def complete(self, job_id, result):
        with self._lock:
            job = self._read(job_id)
            job["status"], job["stage"], job["progress"] = "complete", "complete", 100
            job["result"], job["message"], job["updated_at"] = result, None, utc_now()
            for stage in STAGES:
                job["stages"][stage] = {"state": "done", "progress": 100, "sub": job["stages"][stage].get("sub", "")}
            self._write(job)
            self._emit_locked(job_id, {"type": "job_complete", "job_id": job_id, "status": "complete", "result": result})

    def fail(self, job_id, message, stage=None):
        with self._lock:
            job = self._read(job_id)
            job["status"], job["message"], job["updated_at"] = "error", str(message), utc_now()
            if stage in STAGES:
                job["stage"] = stage
                job["stages"][stage] = {"state": "error", "progress": job["stages"][stage].get("progress", 0), "sub": str(message)}
            self._write(job)
            self._emit_locked(job_id, {"type": "job_error", "job_id": job_id, "stage": stage, "status": "error", "message": str(message)})

    def reset_for_retry(self, job_id):
        with self._lock:
            job = self._read(job_id)
            if job["status"] not in {"error", "complete"}:
                raise ValueError("Only completed or failed jobs can be retried")
            job["status"], job["stage"], job["progress"], job["message"], job["result"] = "queued", "ingest", 0, None, None
            job["stages"] = {stage: {"state": "queued", "progress": 0, "sub": ""} for stage in STAGES}
            job["updated_at"] = utc_now()
            self._write(job)
            self._emit_locked(job_id, {"type": "job_requeued", "job_id": job_id, "status": "queued"})

    def recover_incomplete(self):
        with self._lock:
            for job_file in self.jobs_root.glob("*/job.json"):
                try:
                    job = json.loads(job_file.read_text(encoding="utf-8"))
                    if job.get("status") in {"queued", "running"}:
                        job["status"] = "error"
                        job["message"] = "Processing interrupted by a backend restart; retry the job."
                        job["updated_at"] = utc_now()
                        self._write(job)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue

    def _emit_locked(self, job_id, event):
        event = {**event, "timestamp": utc_now()}
        self._events.setdefault(job_id, []).append(event)
        self._events[job_id] = self._events[job_id][-200:]
        try:
            with paths_for(self.jobs_root, job_id).events_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass
        for loop, queue in list(self._subscribers.get(job_id, set())):
            loop.call_soon_threadsafe(queue.put_nowait, copy.deepcopy(event))

    async def subscribe(self, job_id):
        import asyncio
        with self._lock:
            self._read(job_id)
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()
            subscription = (loop, queue)
            self._subscribers.setdefault(job_id, set()).add(subscription)
            return queue, subscription, copy.deepcopy(self._events.get(job_id, []))

    def unsubscribe(self, job_id, subscription):
        with self._lock:
            self._subscribers.get(job_id, set()).discard(subscription)
