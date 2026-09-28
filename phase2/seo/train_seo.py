"""
ASTERRA AI — S-EO STAGE 2
ROBUST STREAMING FULL-PARAMETER FINE-TUNER

This revision is intended for the observed Windows + Hugging Face remote
streaming failures.

Key fixes:
  1. Automatically resumes from models\asterra_seo\seo_latest.pth.
  2. Keeps the Stage-1 Vaihingen checkpoint as the fallback initializer.
  3. Treats --steps as a MAXIMUM, not a guaranteed dataset length.
  4. Validation defaults to only 10 samples, not 150.
  5. Validation never scans thousands of samples just because a requested
     validation count is unavailable.
  6. Remote RGB multipart .aa/.ab/.ac/.ad objects are concatenated as ONE
     gzip byte stream instead of opening .ab/.ac/.ad as independent gzip
     archives.
  7. Remote HTTP failures are retried and a failed resource is skipped when
     possible.
  8. A broken/short stream ends the current epoch gracefully.
  9. seo_latest.pth is overwritten with the newest completed epoch.
 10. seo_best.pth is overwritten ONLY when validation MAE improves.
 11. Checkpoints are written atomically through a temporary file.
 12. Existing checkpoints are preserved; no reset to Vaihingen occurs when
     seo_latest.pth exists.
 14. DSM is cached locally once and never reconnected remotely during training.
 15. Validation split is persisted to seo_validation_split.json.
 16. NumPy arrays are copied/contiguous before torch conversion.
 13. No global dataset index and no full 176 GB corpus download.

Recommended:
    python .\phase2\seo\train_seo.py --stream-test
    python .\phase2\seo\train_seo.py --validation-stream-test
    python .\phase2\seo\train_seo.py --gpu-smoke-test
    python .\phase2\seo\train_seo.py --steps 20 --val-steps 5
    python .\phase2\seo\train_seo.py

For a real run, the bare command above resumes automatically from
seo_latest.pth when it exists.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import io
import json
import os
import math
import random
import sys
import time
import tarfile
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import webdataset as wds


# ============================================================================
# PROJECT
# ============================================================================

ROOT = Path(r"D:\Asterra AI")

DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"
if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================================
# CHECKPOINTS
# ============================================================================

VAIHINGEN_CHECKPOINT = (
    ROOT / "models" / "asterra_vaihingen" / "vaihingen_best.pth"
)

OUTPUT_DIR = ROOT / "models" / "asterra_seo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_CHECKPOINT = OUTPUT_DIR / "seo_latest.pth"
BEST_CHECKPOINT = OUTPUT_DIR / "seo_best.pth"
HISTORY_FILE = OUTPUT_DIR / "seo_training_history.json"

# Local DSM cache: only the required dsm_max archive is downloaded once.
DSM_CACHE_ROOT = OUTPUT_DIR / "seo_dsm_cache"
DSM_LOCAL_ARCHIVE = DSM_CACHE_ROOT / "dsm_max.tar.gz"
DSM_DOWNLOAD_TMP = DSM_CACHE_ROOT / "dsm_max.tar.gz.part"
DSM_SCENE_DIR = DSM_CACHE_ROOT / "scenes"
DSM_MANIFEST = DSM_CACHE_ROOT / "manifest.json"
VALIDATION_SPLIT_FILE = OUTPUT_DIR / "seo_validation_split.json"


# ============================================================================
# S-EO REMOTE RESOURCES
# ============================================================================

REPO = "https://huggingface.co/datasets/emasquil/shadow-eo/resolve/main"

RGB_PARTS = [
    f"{REPO}/pansharpened_crops_color_corrected.part.tar.gz.aa",
    f"{REPO}/pansharpened_crops_color_corrected.part.tar.gz.ab",
    f"{REPO}/pansharpened_crops_color_corrected.part.tar.gz.ac",
    f"{REPO}/pansharpened_crops_color_corrected.part.tar.gz.ad",
]

DSM_URL = f"{REPO}/dsm_max.tar.gz"


# ============================================================================
# CONFIGURATION
# ============================================================================

PATCH_SIZE = 512
MODEL_SIZE = 518

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4

LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-4

EPOCHS = 3

# IMPORTANT: this is a maximum number of training samples/iterations.
# Remote WebDataset/HTTP streaming is not treated as a fixed-length dataset.
DEFAULT_STEPS = 1500

# Short validation by default. Override with --val-steps.
DEFAULT_VAL_STEPS = 10

AMP_ENABLED = True
GRADIENT_CHECKPOINTING = True
GRAD_CLIP = 1.0

VAL_FRACTION = 0.10

DSM_CACHE_SCENES = 16
MIN_VALID_DSM = 0.80
DSM_MIN = 0.0
DSM_MAX = 500.0

# Split discovery is bounded and does not create a persistent index.
VALIDATION_DISCOVERY_PAIRS = 250
MIN_SCENES_FOR_SCENE_SPLIT = 3

# Validation must not wander through an enormous remote stream.
# It is deliberately bounded relative to the requested number of samples.
MAX_VALIDATION_SCAN_MULTIPLIER = 10
MIN_VALIDATION_SCAN = 30

PRINT_EVERY = 10
SEED = 42

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]

# HTTP streaming retry policy.
HTTP_RETRIES = 3
HTTP_TIMEOUT = 60

# Local DSM cache behavior.
DSM_DOWNLOAD_CHUNK_MB = 8
DSM_LOCAL_LRU_SCENES = 4


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_gpu_info() -> None:
    print("=" * 78)
    print("ASTERRA — S-EO STAGE 2 FULL-PARAMETER FINE-TUNING")
    print("=" * 78)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {props.total_memory / 1024**3:.3f} GB")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("GPU: CUDA NOT AVAILABLE")

    print()
    print("Dataset: emasquil/shadow-eo")
    print("Mode: HTTP/TAR STREAMING")
    print()
    print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Model input: {MODEL_SIZE} x {MODEL_SIZE}")
    print()
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRAD_ACCUMULATION}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUMULATION}")
    print()
    print(f"AMP: {AMP_ENABLED}")
    print(f"Gradient checkpointing: {GRADIENT_CHECKPOINTING}")
    print()
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Epochs: {EPOCHS}")
    print(f"Max steps/epoch: {DEFAULT_STEPS}")
    print(f"Validation samples/epoch: {DEFAULT_VAL_STEPS}")
    print()


def print_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return

    print(f"--- GPU MEMORY: {label} ---")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.3f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved()/1024**3:.3f} GB")
    print(f"Peak:      {torch.cuda.max_memory_allocated()/1024**3:.3f} GB")
    print()


def scene_id_from_key(key: str) -> str:
    key = str(key).strip().replace("\\", "/")
    parts = [p for p in key.split("/") if p]

    for p in parts:
        if p.startswith(("OMA_", "UCSD_")):
            return p

    for prefix in ("OMA_", "UCSD_"):
        pos = key.find(prefix)
        if pos >= 0:
            rest = key[pos:].split("/")[0]
            chunks = rest.split("_")
            if len(chunks) >= 2:
                return "_".join(chunks[:2])

    return ""


def stable_fraction(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def scene_is_validation(scene: str) -> bool:
    return stable_fraction(f"scene::{scene}") < VAL_FRACTION


def key_is_validation(scene: str, key: str) -> bool:
    return stable_fraction(f"key::{scene}::{key}") < VAL_FRACTION


# ============================================================================
# VALIDATION POLICY
# ============================================================================

class ValidationPolicy:
    def __init__(
        self,
        mode: str,
        validation_scenes: Optional[set[str]] = None,
    ):
        if mode not in {"scene", "key"}:
            raise ValueError("mode must be 'scene' or 'key'")
        self.mode = mode
        self.validation_scenes = validation_scenes or set()

    def is_validation(self, scene: str, key: str) -> bool:
        if self.mode == "scene":
            return scene in self.validation_scenes
        return key_is_validation(scene, key)

    def describe(self) -> str:
        if self.mode == "scene":
            return (
                f"SCENE-LEVEL ({len(self.validation_scenes)} held-out scenes)"
            )
        return "KEY-LEVEL FALLBACK (deterministic crop split)"


def _policy_from_data(data: Dict[str, Any]) -> Optional[ValidationPolicy]:
    mode = data.get("mode")
    scenes = data.get("validation_scenes", [])
    if mode == "scene" and scenes:
        return ValidationPolicy(mode="scene", validation_scenes=set(map(str, scenes)))
    if mode == "key":
        return ValidationPolicy(mode="key")
    return None


def _load_persistent_validation_policy() -> Optional[ValidationPolicy]:
    if VALIDATION_SPLIT_FILE.exists():
        try:
            data = json.loads(VALIDATION_SPLIT_FILE.read_text(encoding="utf-8"))
            policy = _policy_from_data(data)
            if policy is not None:
                print("[OK] Reusing persistent validation split:")
                print(f"     {VALIDATION_SPLIT_FILE}")
                return policy
        except Exception as exc:
            print(f"[WARNING] Validation split file unreadable: {type(exc).__name__}: {exc}")

    # Existing history is small and can preserve the exact split from previous runs.
    history = load_history()
    for item in reversed(history):
        validation = item.get("validation", {}) if isinstance(item, dict) else {}
        mode = validation.get("validation_mode")
        scenes = validation.get("validation_scenes", [])
        if mode == "scene" and scenes:
            policy = ValidationPolicy(mode="scene", validation_scenes=set(map(str, scenes)))
            _save_validation_policy(policy, source="training history")
            return policy
        if mode == "key":
            policy = ValidationPolicy(mode="key")
            _save_validation_policy(policy, source="training history")
            return policy

    # Checkpoint metadata is used only if history did not contain the split.
    if LATEST_CHECKPOINT.exists():
        try:
            payload = torch.load(LATEST_CHECKPOINT, map_location="cpu", weights_only=False)
            config = payload.get("config", {}) if isinstance(payload, dict) else {}
            mode = config.get("validation_mode")
            scenes = config.get("validation_scenes", [])
            if mode == "scene" and scenes:
                policy = ValidationPolicy(mode="scene", validation_scenes=set(map(str, scenes)))
                _save_validation_policy(policy, source="seo_latest.pth metadata")
                del payload
                gc.collect()
                return policy
            if mode == "key":
                policy = ValidationPolicy(mode="key")
                _save_validation_policy(policy, source="seo_latest.pth metadata")
                del payload
                gc.collect()
                return policy
            del payload
            gc.collect()
        except Exception as exc:
            print(f"[WARNING] Could not read validation metadata from checkpoint: {type(exc).__name__}: {exc}")

    return None


def _save_validation_policy(policy: ValidationPolicy, source: str = "discovery") -> None:
    data = {
        "version": 1,
        "source": source,
        "mode": policy.mode,
        "validation_scenes": sorted(policy.validation_scenes),
        "created_at": time.time(),
    }
    tmp = VALIDATION_SPLIT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, VALIDATION_SPLIT_FILE)


def discover_validation_policy(scan_pairs: int = VALIDATION_DISCOVERY_PAIRS) -> Tuple[ValidationPolicy, int]:
    persistent = _load_persistent_validation_policy()
    if persistent is not None:
        print(f"Validation mode: {persistent.describe()}")
        if persistent.validation_scenes:
            print(f"Held-out scenes: {sorted(persistent.validation_scenes)}")
        return persistent, 0

    print()
    print("=" * 78)
    print("S-EO VALIDATION SPLIT DISCOVERY (ONE-TIME)")
    print("=" * 78)
    print(f"Scanning at most {scan_pairs} paired samples once; result will be persisted.")

    scenes: set[str] = set()
    scanned = 0
    iterator = make_pair_stream(max_pairs=scan_pairs)
    for scene, key, rgb, dsm in iterator:
        scanned += 1
        prepared = prepare_sample(scene, key, rgb, dsm)
        if prepared is not None:
            scenes.add(scene)
        if len(scenes) >= MIN_SCENES_FOR_SCENE_SPLIT:
            break

    if len(scenes) >= MIN_SCENES_FOR_SCENE_SPLIT:
        val_scenes = {scene for scene in scenes if scene_is_validation(scene)}
        if not val_scenes:
            ordered = sorted(scenes, key=lambda s: stable_fraction(f"force::{s}"))
            val_scenes = {ordered[-1]}
        policy = ValidationPolicy(mode="scene", validation_scenes=val_scenes)
    else:
        print("[WARNING] Too few scenes for reliable scene-level validation.")
        print("[OK] Using deterministic KEY-LEVEL fallback.")
        policy = ValidationPolicy(mode="key")

    _save_validation_policy(policy, source="one-time bounded discovery")
    print(f"Distinct scenes discovered: {len(scenes)}")
    print(f"Validation mode: {policy.describe()}")
    if policy.validation_scenes:
        print(f"Validation scenes: {sorted(policy.validation_scenes)}")
    print(f"[OK] Persistent split saved: {VALIDATION_SPLIT_FILE}")
    return policy, scanned


# ============================================================================
# HTTP/TAR STREAMING
# ============================================================================

class ConcatenatedHTTPReader:
    """
    File-like reader which concatenates multiple HTTP objects.

    The S-EO RGB .aa/.ab/.ac/.ad objects are pieces of ONE gzip stream.
    They must therefore be concatenated at the byte level before gzip/tar
    decoding. Opening .ab/.ac/.ad independently produces the observed
    'invalid header' errors.

    Only the bytes currently consumed by gzip/tar are kept in memory.
    """

    def __init__(
        self,
        urls: List[str],
        retries: int = HTTP_RETRIES,
        timeout: int = HTTP_TIMEOUT,
    ):
        self.urls = urls
        self.retries = retries
        self.timeout = timeout
        self.index = 0
        self.response = None
        self.closed = False

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def writable(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True
        if self.response is not None:
            try:
                self.response.close()
            except Exception:
                pass
            self.response = None

    def _open_next(self) -> bool:
        if self.index >= len(self.urls):
            return False

        url = self.urls[self.index]
        self.index += 1

        last_exc = None

        for attempt in range(1, self.retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "ASTERRA-S-EO-Trainer/1.0",
                        "Accept": "*/*",
                    },
                )
                self.response = urllib.request.urlopen(
                    request,
                    timeout=self.timeout,
                )
                return True
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(min(2 * attempt, 5))

        print()
        print(f"[WARNING] HTTP resource failed after {self.retries} tries:")
        print(f"  {url}")
        print(f"  {type(last_exc).__name__}: {last_exc}")
        print("  Continuing with the next resource.")
        return self._open_next()

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            return b""

        chunks = []
        remaining = size

        while True:
            if self.response is None:
                if not self._open_next():
                    break

            try:
                want = 1024 * 1024 if remaining < 0 else max(
                    1, min(remaining, 1024 * 1024)
                )
                data = self.response.read(want)

                if data:
                    chunks.append(data)
                    if remaining >= 0:
                        remaining -= len(data)
                        if remaining <= 0:
                            break
                    continue

                try:
                    self.response.close()
                except Exception:
                    pass
                self.response = None

            except Exception as exc:
                url = self.urls[max(0, self.index - 1)]
                print()
                print("[WARNING] HTTP stream read failed:")
                print(f"  {url}")
                print(f"  {type(exc).__name__}: {exc}")
                print("  Skipping to the next resource.")

                try:
                    self.response.close()
                except Exception:
                    pass
                self.response = None

                # The current multipart piece cannot safely be resumed in
                # the middle because gzip byte continuity matters. Skip the
                # rest of the multipart stream rather than emitting corrupt
                # bytes. For a single DSM tar this ends that resource.
                break

            if remaining == 0:
                break

        return b"".join(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _pil_from_tar_member(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> Optional[Image.Image]:
    extracted = tar.extractfile(member)
    if extracted is None:
        return None

    try:
        raw = extracted.read()
        return Image.open(io.BytesIO(raw)).copy()
    except Exception:
        return None
    finally:
        try:
            extracted.close()
        except Exception:
            pass


def _iter_tar_images(
    urls: List[str],
) -> Iterator[Tuple[str, Image.Image]]:
    """
    Stream image files from one logical gzip/tar resource.
    """

    source = ConcatenatedHTTPReader(urls)

    try:
        # r|gz is explicitly streaming and never builds a full archive in RAM.
        with tarfile.open(fileobj=source, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                lower = member.name.lower()

                if not lower.endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
                ):
                    continue

                image = _pil_from_tar_member(tar, member)
                if image is None:
                    continue

                yield member.name, image

    except (tarfile.TarError, EOFError, OSError, gzip.BadGzipFile) as exc:
        print()
        print("[WARNING] TAR/GZIP stream ended or was damaged:")
        print(f"  {type(exc).__name__}: {exc}")
        print("  The current epoch/test will continue with samples already read.")
    finally:
        source.close()


def iter_rgb_samples() -> Iterator[Tuple[str, str, np.ndarray]]:
    """
    RGB .aa/.ab/.ac/.ad are treated as one logical multipart gzip stream.
    """

    try:
        for key, image in _iter_tar_images(RGB_PARTS):
            scene = scene_id_from_key(key)
            if not scene:
                continue

            try:
                rgb = np.asarray(image.convert("RGB"))
                rgb = np.ascontiguousarray(rgb)
            except Exception:
                continue

            yield scene, key, rgb

    except Exception as exc:
        print()
        print("[WARNING] RGB streaming failed:")
        print(f"  {type(exc).__name__}: {exc}")
        print("  Continuing with whatever samples were already produced.")


def _download_dsm_archive() -> Path:
    """Download dsm_max.tar.gz exactly once into the S-EO output cache."""
    DSM_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    DSM_SCENE_DIR.mkdir(parents=True, exist_ok=True)

    if DSM_LOCAL_ARCHIVE.exists() and DSM_LOCAL_ARCHIVE.stat().st_size > 1024:
        return DSM_LOCAL_ARCHIVE

    if DSM_DOWNLOAD_TMP.exists():
        try:
            DSM_DOWNLOAD_TMP.unlink()
        except OSError:
            pass

    print()
    print("[INFO] Downloading required DSM archive once:")
    print(f"       {DSM_URL}")
    print(f"       -> {DSM_LOCAL_ARCHIVE}")
    print("       RGB remains streamed remotely; the 176 GB corpus is NOT downloaded.")

    request = urllib.request.Request(
        DSM_URL,
        headers={
            "User-Agent": "ASTERRA-S-EO-Trainer/2.0",
            "Accept": "*/*",
        },
    )

    started = time.time()
    total = 0
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response, open(DSM_DOWNLOAD_TMP, "wb") as out:
        while True:
            chunk = response.read(DSM_DOWNLOAD_CHUNK_MB * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            elapsed = max(time.time() - started, 1e-6)
            mb = total / (1024 ** 2)
            speed = mb / elapsed
            print(f"\r[DSM] {mb:,.0f} MB downloaded | {speed:.1f} MB/s", end="", flush=True)

    print()
    if total <= 1024:
        raise RuntimeError("DSM download produced an unexpectedly small file.")

    os.replace(DSM_DOWNLOAD_TMP, DSM_LOCAL_ARCHIVE)
    print(f"[OK] DSM archive cached locally ({total / (1024**3):.2f} GB).")
    return DSM_LOCAL_ARCHIVE


def _build_dsm_scene_cache(archive: Path) -> List[str]:
    """Extract only DSM image members into per-scene .npy files once."""
    DSM_SCENE_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(DSM_SCENE_DIR.glob("*.npy"))
    if existing:
        scenes = [p.stem for p in existing]
        print(f"[OK] Reusing local DSM scene cache: {len(scenes)} scenes.")
        return scenes

    print()
    print("[INFO] Building local DSM scene cache (one-time operation)...")
    scenes = set()
    count = 0
    started = time.time()

    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                lower = member.name.lower()
                if not lower.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp")):
                    continue
                scene = scene_id_from_key(member.name)
                if not scene:
                    continue
                target = DSM_SCENE_DIR / f"{scene}.npy"
                if target.exists():
                    scenes.add(scene)
                    continue
                image = _pil_from_tar_member(tar, member)
                if image is None:
                    continue
                arr = np.asarray(image, dtype=np.float32)
                arr = np.ascontiguousarray(arr)
                if arr.ndim == 3:
                    if arr.shape[0] == 1:
                        arr = arr[0]
                    elif arr.shape[-1] == 1:
                        arr = arr[..., 0]
                    else:
                        arr = arr[..., 0]
                if arr.ndim != 2:
                    continue
                np.save(target, arr, allow_pickle=False)
                scenes.add(scene)
                count += 1
                if count % 10 == 0:
                    print(f"\r[DSM cache] {count} scene rasters extracted | {time.time()-started:.1f}s", end="", flush=True)
    except (tarfile.TarError, EOFError, OSError, gzip.BadGzipFile) as exc:
        print()
        print(f"[ERROR] Local DSM archive is incomplete/corrupt: {type(exc).__name__}: {exc}")
        print("[INFO] Delete the partial DSM cache and rerun so it can be downloaded again.")
        raise

    print()
    if not scenes:
        raise RuntimeError("No usable DSM scene rasters were found in the local archive.")

    manifest = {
        "version": 1,
        "archive": str(archive),
        "scenes": sorted(scenes),
        "created_at": time.time(),
    }
    DSM_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Local DSM scene cache ready: {len(scenes)} scenes.")
    return sorted(scenes)


def ensure_local_dsm_cache() -> List[str]:
    """Ensure DSM is local and scene-addressable; no network DSM access thereafter."""
    archive = _download_dsm_archive()
    return _build_dsm_scene_cache(archive)


def iter_dsm_samples() -> Iterator[Tuple[str, np.ndarray]]:
    """Read DSM scene rasters from the persistent local cache."""
    if not DSM_SCENE_DIR.exists():
        ensure_local_dsm_cache()
    for path in sorted(DSM_SCENE_DIR.glob("*.npy")):
        try:
            scene = path.stem
            dsm = np.load(path, mmap_mode="r")
            yield scene, np.ascontiguousarray(dsm)
        except Exception as exc:
            print(f"[WARNING] Could not read local DSM scene {path.name}: {type(exc).__name__}: {exc}")


class SEOStreamingPairs:
    """Stream RGB remotely and pair each RGB crop with a local DSM scene raster."""

    def __init__(self, max_pairs: int = 0, cache_scenes: int = DSM_LOCAL_LRU_SCENES):
        self.max_pairs = max_pairs
        self.cache_scenes = cache_scenes

    def __iter__(self):
        rgb_iter = iter_rgb_samples()
        cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        produced = 0

        for rgb_scene, rgb_key, rgb in rgb_iter:
            dsm = cache.get(rgb_scene)
            if dsm is None:
                path = DSM_SCENE_DIR / f"{rgb_scene}.npy"
                if not path.exists():
                    continue
                try:
                    dsm = np.load(path, mmap_mode="r")
                    dsm = np.ascontiguousarray(dsm)
                except Exception:
                    continue
                cache[rgb_scene] = dsm
                cache.move_to_end(rgb_scene)
                while len(cache) > self.cache_scenes:
                    cache.popitem(last=False)
            else:
                cache.move_to_end(rgb_scene)

            yield rgb_scene, rgb_key, rgb, dsm
            produced += 1
            if self.max_pairs > 0 and produced >= self.max_pairs:
                return


def make_pair_stream(max_pairs: int = 0) -> Iterator[Tuple[str, str, np.ndarray, np.ndarray]]:
    # DSM must already be local before a stream is opened.
    if not DSM_SCENE_DIR.exists() or not any(DSM_SCENE_DIR.glob("*.npy")):
        ensure_local_dsm_cache()
    return iter(SEOStreamingPairs(max_pairs=max_pairs))


# ============================================================================
# ARRAY / SAMPLE PREPARATION
# ============================================================================

def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    rgb = np.asarray(rgb)

    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)

    if rgb.ndim != 3:
        raise ValueError(f"Invalid RGB shape: {rgb.shape}")

    if rgb.shape[-1] != 3:
        if rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        else:
            raise ValueError(f"RGB must have 3 channels: {rgb.shape}")

    image = Image.fromarray(
        np.clip(rgb, 0, 255).astype(np.uint8),
        mode="RGB",
    )

    image = image.resize(
        (size, size),
        resample=Image.Resampling.BILINEAR,
    )

    return np.asarray(image)


def resize_dsm(dsm: np.ndarray, size: int) -> np.ndarray:
    dsm = np.asarray(dsm)

    if dsm.ndim == 3:
        if dsm.shape[0] == 1:
            dsm = dsm[0]
        elif dsm.shape[-1] == 1:
            dsm = dsm[..., 0]
        else:
            dsm = dsm[..., 0]

    if dsm.ndim != 2:
        raise ValueError(
            f"Invalid DSM shape after conversion: {dsm.shape}"
        )

    dsm = np.asarray(dsm, dtype=np.float32)

    tensor = torch.from_numpy(
        np.ascontiguousarray(dsm)
    )[None, None]

    tensor = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )

    return np.ascontiguousarray(
        tensor[0, 0].cpu().numpy(),
        dtype=np.float32,
    )


def clean_dsm(
    dsm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    dsm = np.asarray(dsm, dtype=np.float32)

    valid = np.isfinite(dsm)
    valid &= dsm > DSM_MIN
    valid &= dsm < DSM_MAX

    clean = np.where(
        valid,
        dsm,
        0.0,
    ).astype(np.float32)

    return clean, valid.astype(np.float32)


def normalize_rgb(rgb: np.ndarray) -> torch.Tensor:
    rgb = np.asarray(rgb)

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    rgb = np.clip(
        rgb,
        0,
        255,
    ).astype(np.uint8)

    tensor = torch.from_numpy(
        np.ascontiguousarray(rgb).copy()
    )

    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(
            f"Unexpected RGB tensor shape: {tensor.shape}"
        )

    return tensor.permute(2, 0, 1).float() / 255.0


def prepare_sample(
    scene: str,
    key: str,
    rgb: np.ndarray,
    dsm: np.ndarray,
) -> Optional[Dict[str, Any]]:
    try:
        rgb = resize_rgb(rgb, PATCH_SIZE)
        dsm = resize_dsm(dsm, PATCH_SIZE)

        dsm, mask = clean_dsm(dsm)

        valid_fraction = float(mask.mean())

        if valid_fraction < MIN_VALID_DSM:
            return None

        return {
            "scene": scene,
            "key": key,
            "image": normalize_rgb(rgb),
            "depth": torch.from_numpy(
                np.ascontiguousarray(dsm)
            ).float(),
            "mask": torch.from_numpy(
                np.ascontiguousarray(mask)
            ).float(),
            "valid_fraction": valid_fraction,
        }

    except Exception:
        return None


# ============================================================================
# MODEL
# ============================================================================

def _extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        for candidate in (
            "model_state_dict",
            "state_dict",
            "model",
            "weights",
        ):
            value = checkpoint.get(candidate)
            if isinstance(value, dict):
                return value

        # A raw state_dict is also a dict.
        return checkpoint

    raise RuntimeError(
        "Checkpoint does not contain a usable state dictionary."
    )


def _clean_state_dict(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    cleaned = {}

    for key, value in state.items():
        new_key = str(key)

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned[new_key] = value

    return cleaned


def create_model(
    initial_checkpoint: Path = VAIHINGEN_CHECKPOINT,
) -> nn.Module:
    print()
    print("=" * 78)
    print("CREATING DEPTH ANYTHING V2 LARGE")
    print("=" * 78)

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    if not initial_checkpoint.exists():
        raise FileNotFoundError(
            "Initial checkpoint not found:\n"
            f"{initial_checkpoint}"
        )

    print()
    if initial_checkpoint == LATEST_CHECKPOINT:
        print("Loading existing S-EO checkpoint (resume weights):")
    else:
        print("Loading Stage-1 Vaihingen checkpoint:")
    print(initial_checkpoint)

    checkpoint = torch.load(
        initial_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    state = _extract_state_dict(checkpoint)
    state = _clean_state_dict(state)

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    # If this is the S-EO latest checkpoint, keep only non-model metadata
    # so resume_from_latest can restore optimizer/scaler without loading the
    # 1.4 GB checkpoint a second time.
    if initial_checkpoint == LATEST_CHECKPOINT and isinstance(checkpoint, dict):
        checkpoint = dict(checkpoint)
        checkpoint.pop("model_state_dict", None)
        checkpoint.pop("state_dict", None)
        checkpoint.pop("model", None)
        checkpoint.pop("weights", None)
        setattr(model, "_seo_resume_payload", checkpoint)

    print(f"Missing keys:    {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing or unexpected:
        for key in missing[:20]:
            print(f"  Missing: {key}")
        for key in unexpected[:20]:
            print(f"  Unexpected: {key}")

        raise RuntimeError(
            "Initial checkpoint is not fully compatible with "
            "Depth Anything V2 Large."
        )

    print("[OK] Vaihingen checkpoint loaded perfectly.")
    return model


def load_model_only(
    model: nn.Module,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """
    Load an existing S-EO checkpoint into an already-created model.
    """

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state = _extract_state_dict(payload)
    state = _clean_state_dict(state)

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "Existing S-EO checkpoint is incompatible with the current "
            "Depth Anything V2 model.\n"
            f"Missing: {len(missing)}\n"
            f"Unexpected: {len(unexpected)}"
        )

    print(f"[OK] Resumed model weights from: {checkpoint_path}")
    return payload if isinstance(payload, dict) else {}


def enable_gradient_checkpointing(
    model: nn.Module,
) -> None:
    if not GRADIENT_CHECKPOINTING:
        return

    print()
    print("Enabling gradient checkpointing...")

    pretrained = getattr(model, "pretrained", None)
    blocks = getattr(pretrained, "blocks", None)

    if blocks is None:
        print(
            "[WARNING] Transformer blocks not found; "
            "continuing without block flag."
        )
        return

    for block in blocks:
        if hasattr(block, "gradient_checkpointing"):
            block.gradient_checkpointing = True

    if hasattr(pretrained, "gradient_checkpointing"):
        pretrained.gradient_checkpointing = True

    print(
        f"Gradient checkpointing enabled for {len(blocks)} "
        "transformer blocks."
    )


def parameter_report(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print("=" * 78)
    print("PARAMETER CONFIGURATION")
    print("=" * 78)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(
        f"Trainable percentage: "
        f"{100.0 * trainable / max(total, 1):.2f}%"
    )

    if trainable != total:
        raise RuntimeError(
            "Full-parameter fine-tuning requested but not all parameters "
            "are trainable."
        )


# ============================================================================
# MODEL FORWARD / LOSS
# ============================================================================

def model_forward(
    model: nn.Module,
    image: torch.Tensor,
) -> torch.Tensor:
    output = model(image)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError(
            f"Model output is not a tensor: {type(output)}"
        )

    if output.ndim == 3:
        output = output.unsqueeze(1)

    if output.ndim != 4:
        raise ValueError(
            f"Unexpected model output shape: {tuple(output.shape)}"
        )

    if output.shape[1] != 1:
        output = output[:, :1]

    return output


def resize_target_and_mask(
    target: torch.Tensor,
    mask: torch.Tensor,
    size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if target.ndim == 3:
        target = target.unsqueeze(1)

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    target = F.interpolate(
        target.float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    mask = F.interpolate(
        mask.float(),
        size=size,
        mode="nearest",
    )

    return target, mask


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim == 4:
        prediction = prediction[:, 0]
    if target.ndim == 4:
        target = target[:, 0]
    if mask.ndim == 4:
        mask = mask[:, 0]

    valid = (
        (mask > 0.5)
        & torch.isfinite(prediction)
        & torch.isfinite(target)
    )

    if int(valid.sum().item()) == 0:
        raise RuntimeError(
            "Sample contains zero valid pixels after resize."
        )

    return torch.abs(
        prediction.float() - target.float()
    )[valid].mean()


def prepare_for_model(
    sample: Dict[str, Any],
    dev: torch.device,
):
    image = sample["image"].unsqueeze(0).to(
        dev,
        non_blocking=True,
    )
    target = sample["depth"].unsqueeze(0).to(
        dev,
        non_blocking=True,
    )
    mask = sample["mask"].unsqueeze(0).to(
        dev,
        non_blocking=True,
    )

    image = F.interpolate(
        image,
        size=(MODEL_SIZE, MODEL_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return image, target, mask


def autocast_context():
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(
            torch.cuda.is_available()
            and AMP_ENABLED
        ),
    )


def create_scaler():
    if not torch.cuda.is_available() or not AMP_ENABLED:
        return None

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
    except TypeError:
        return torch.cuda.amp.GradScaler(
            enabled=True,
        )


def create_optimizer(model: nn.Module):
    # Adafactor is not guaranteed to be exposed by torch.optim on all
    # installations. Prefer it when available, otherwise use AdamW.
    if hasattr(torch.optim, "Adafactor"):
        print()
        print("Optimizer: torch.optim.Adafactor")
        print("Reason: memory-efficient full-parameter optimization")

        return torch.optim.Adafactor(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    print()
    print("Optimizer: torch.optim.AdamW")
    print(
        "Reason: Adafactor is unavailable in this PyTorch installation."
    )

    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


# ============================================================================
# SAMPLE SELECTION
# ============================================================================

def next_prepared_sample(
    iterator,
    policy: ValidationPolicy,
    want_validation: bool,
    max_scan: int,
) -> Optional[Dict[str, Any]]:
    scanned = 0

    while scanned < max_scan:
        try:
            scene, key, rgb, dsm = next(iterator)
        except StopIteration:
            return None

        scanned += 1

        if policy.is_validation(scene, key) != want_validation:
            continue

        prepared = prepare_sample(
            scene,
            key,
            rgb,
            dsm,
        )

        if prepared is not None:
            return prepared

    return None


# ============================================================================
# STREAM TEST
# ============================================================================

def stream_test(
    num_pairs: int = 10,
) -> bool:
    print()
    print("=" * 78)
    print("S-EO STREAM PAIR TEST")
    print("=" * 78)
    print("Reading only a small number of RGB ↔ DSM pairs.")
    print("No global index is created.")
    print()

    iterator = make_pair_stream(max_pairs=num_pairs)

    count = 0

    for _ in range(num_pairs):
        try:
            scene, key, rgb, dsm = next(iterator)
        except StopIteration:
            break

        sample = prepare_sample(
            scene,
            key,
            rgb,
            dsm,
        )

        if sample is None:
            continue

        count += 1

        print(
            f"PAIR {count:02d}: "
            f"scene={scene} | "
            f"RGB={rgb.shape} | "
            f"DSM={dsm.shape} | "
            f"valid={sample['valid_fraction']:.3f}"
        )

    print()
    print(f"Usable pairs: {count}")

    if count == 0:
        print("[ERROR] No usable RGB ↔ DSM pairs were produced.")
        return False

    print("[OK] S-EO RGB ↔ DSM streaming pair test passed.")
    return True


# ============================================================================
# VALIDATION STREAM TEST
# ============================================================================

def validation_stream_test(
    target: int = 5,
    scan_pairs: int = VALIDATION_DISCOVERY_PAIRS,
) -> bool:
    print()
    print("=" * 78)
    print("S-EO VALIDATION STREAM TEST")
    print("=" * 78)
    print()
    print(
        "This test verifies that validation can obtain REAL samples "
        "without a global dataset index."
    )

    policy, discovery_scanned = discover_validation_policy(
        scan_pairs=scan_pairs,
    )

    print()
    print(f"Validation mode: {policy.describe()}")
    print(f"Discovery pairs scanned: {discovery_scanned}")

    iterator = make_pair_stream(max_pairs=0)

    found = 0
    scanned = 0
    seen: set[str] = set()
    max_scan = max(
        target * MAX_VALIDATION_SCAN_MULTIPLIER,
        MIN_VALIDATION_SCAN,
    )

    start = time.time()

    while found < target and scanned < max_scan:
        try:
            scene, key, rgb, dsm = next(iterator)
        except StopIteration:
            break

        scanned += 1

        if not policy.is_validation(scene, key):
            continue

        identity = f"{scene}::{key}"

        if identity in seen:
            continue

        sample = prepare_sample(
            scene,
            key,
            rgb,
            dsm,
        )

        if sample is None:
            continue

        seen.add(identity)
        found += 1

        print(
            f"VAL PAIR {found:02d}: "
            f"scene={scene} | "
            f"key={key} | "
            f"valid={sample['valid_fraction']:.3f}"
        )

    print()
    print(f"Validation samples found: {found}")
    print(f"Validation pairs scanned: {scanned}")
    print(f"Elapsed: {time.time() - start:.1f}s")

    if found < target:
        print()
        print(
            "[WARNING] Remote stream ended before the requested "
            "validation count."
        )

        # A non-zero real validation set is still a successful stream test.
        if found > 0:
            print(
                "[OK] Validation stream produced real samples; "
                "short stream accepted."
            )
            return True

        print("[ERROR] No real validation samples were produced.")
        return False

    print()
    print("[OK] Validation stream test passed.")
    return True


# ============================================================================
# GPU SMOKE TEST
# ============================================================================

def gpu_smoke_test() -> bool:
    print()
    print("=" * 78)
    print("S-EO GPU ONE-BATCH TEST")
    print("=" * 78)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available.")
        return False

    iterator = make_pair_stream(max_pairs=20)

    sample = None

    for _ in range(20):
        try:
            scene, key, rgb, dsm = next(iterator)
        except StopIteration:
            break

        sample = prepare_sample(
            scene,
            key,
            rgb,
            dsm,
        )

        if sample is not None:
            break

    if sample is None:
        print("[ERROR] No usable S-EO sample available.")
        return False

    dev = torch.device("cuda")

    model = None
    optimizer = None
    scaler = None

    try:
        model = create_model()
        enable_gradient_checkpointing(model)
        parameter_report(model)
        model.to(dev)

        optimizer = create_optimizer(model)
        scaler = create_scaler()

        optimizer.zero_grad(set_to_none=True)

        image, target, mask = prepare_for_model(
            sample,
            dev,
        )

        print()
        print(
            f"Smoke sample: scene={sample['scene']} "
            f"key={sample['key']}"
        )

        with autocast_context():
            prediction = model_forward(
                model,
                image,
            )

            target, mask = resize_target_and_mask(
                target,
                mask,
                prediction.shape[-2:],
            )

            loss = masked_l1(
                prediction,
                target,
                mask,
            )

            scaled_loss = loss / GRAD_ACCUMULATION

        print(
            f"Forward loss: "
            f"{float(loss.detach().cpu()):.6f}"
        )

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
            scaler.unscale_(optimizer)
        else:
            scaled_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        print("[OK] Forward + backward + optimizer step succeeded.")
        print_memory("after GPU smoke test")

        return True

    except torch.cuda.OutOfMemoryError:
        print()
        print("[ERROR] CUDA out of memory during GPU smoke test.")
        print(
            "The current full-parameter 512→518 configuration does not "
            "fit the current GPU memory state."
        )
        return False

    except Exception as exc:
        print()
        print(
            f"[ERROR] GPU smoke test failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return False

    finally:
        del model
        del optimizer
        del scaler
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# TRAINING
# ============================================================================

def _optimizer_has_gradients(model: nn.Module) -> bool:
    """Return True only when at least one trainable parameter has a gradient."""
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            return True
    return False


def _safe_optimizer_step(model: nn.Module, optimizer, scaler) -> bool:
    """
    Safely perform an optimizer update.

    Prevents the PyTorch GradScaler failure:
        AssertionError: No inf checks were recorded for this optimizer.
    """
    if not _optimizer_has_gradients(model):
        optimizer.zero_grad(set_to_none=True)
        return False

    if scaler is not None:
        scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
    )

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    return True


def train_steps(
    model: nn.Module,
    optimizer,
    scaler,
    iterator,
    policy: ValidationPolicy,
    dev: torch.device,
    steps: int,
    epoch: int,
) -> Tuple[float, int, bool]:
    """
    Train for at most `steps` usable samples.

    Remote stream exhaustion is a normal recoverable condition.
    Partial gradient accumulation is flushed safely.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss_sum = 0.0
    successful = 0
    exhausted = False
    accumulation_count = 0
    optimizer_updates = 0

    start = time.time()

    for step in range(1, steps + 1):

        sample = next_prepared_sample(
            iterator,
            policy=policy,
            want_validation=False,
            max_scan=500,
        )

        if sample is None:
            exhausted = True
            print()
            print(
                "[WARNING] Training stream ended before the requested "
                f"{steps} samples."
            )
            print(
                f"[INFO] Completed {successful}/{steps} requested "
                "training samples this epoch."
            )
            break

        image, target, mask = prepare_for_model(
            sample,
            dev,
        )

        try:
            with autocast_context():
                prediction = model_forward(
                    model,
                    image,
                )

                target_resized, mask_resized = (
                    resize_target_and_mask(
                        target,
                        mask,
                        prediction.shape[-2:],
                    )
                )

                loss = masked_l1(
                    prediction,
                    target_resized,
                    mask_resized,
                )

                if not torch.isfinite(loss):
                    print(
                        f"[WARNING] Non-finite loss at step {step}; "
                        "skipping this sample."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accumulation_count = 0
                    continue

                scaled_loss = loss / GRAD_ACCUMULATION

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accumulation_count += 1

            should_step = (
                accumulation_count >= GRAD_ACCUMULATION
                or step == steps
            )

            if should_step:
                if _safe_optimizer_step(
                    model,
                    optimizer,
                    scaler,
                ):
                    optimizer_updates += 1
                accumulation_count = 0

            value = float(loss.detach().float().cpu())
            loss_sum += value
            successful += 1

        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            print()
            print(
                "[ERROR] CUDA OOM during training step. "
                "The current sample was not counted."
            )
            raise

        except (RuntimeError, AssertionError) as exc:
            if "No inf checks were recorded for this optimizer" in str(exc):
                print()
                print(
                    "[WARNING] GradScaler found no optimizer gradients "
                    "at this boundary."
                )
                print(
                    "[INFO] Clearing partial gradients and continuing."
                )
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                continue
            raise

        if (
            step == 1
            or step % PRINT_EVERY == 0
            or step == steps
        ):
            elapsed = time.time() - start
            average = loss_sum / max(successful, 1)

            print(
                f"Epoch {epoch} "
                f"Step {step}/{steps} | "
                f"Loss {value:.5f} | "
                f"Avg {average:.5f} | "
                f"Scene {sample['scene']} | "
                f"Updates {optimizer_updates} | "
                f"Time {elapsed:.1f}s"
            )

            if step == 1:
                print_memory(
                    f"epoch {epoch} step {step}"
                )

    # Flush a final partial accumulation safely if the remote stream
    # ended before reaching the requested number of samples.
    if accumulation_count > 0:
        _safe_optimizer_step(
            model,
            optimizer,
            scaler,
        )

    optimizer.zero_grad(set_to_none=True)

    if successful == 0:
        raise RuntimeError(
            "Zero successful training samples were produced."
        )

    return (
        loss_sum / successful,
        successful,
        exhausted,
    )


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
@torch.no_grad()
def validate(
    model: nn.Module,
    iterator,
    policy: ValidationPolicy,
    dev: torch.device,
    requested_samples: int,
) -> Dict[str, Any]:
    """
    Short bounded validation.

    Remote exhaustion is recoverable. Zero validation samples are marked
    unavailable and NEVER interpreted as a perfect metric.
    """

    model.eval()

    total_abs = 0.0
    total_sq = 0.0
    valid_pixels = 0
    samples = 0
    scanned = 0

    max_scan = max(
        requested_samples * MAX_VALIDATION_SCAN_MULTIPLIER,
        MIN_VALIDATION_SCAN,
    )

    start = time.time()

    print()
    print(f"Validation mode: {policy.describe()}")
    print(
        f"Validation target: {requested_samples} samples "
        f"(maximum scan {max_scan})"
    )

    while (
        samples < requested_samples
        and scanned < max_scan
    ):
        try:
            scene, key, rgb, dsm = next(iterator)
        except StopIteration:
            break

        scanned += 1

        if not policy.is_validation(scene, key):
            continue

        sample = prepare_sample(
            scene,
            key,
            rgb,
            dsm,
        )

        if sample is None:
            continue

        try:
            image, target, mask = prepare_for_model(
                sample,
                dev,
            )

            with autocast_context():
                prediction = model_forward(
                    model,
                    image,
                )

                target_resized, mask_resized = (
                    resize_target_and_mask(
                        target,
                        mask,
                        prediction.shape[-2:],
                    )
                )

            prediction = prediction.float()

            if prediction.ndim == 4:
                prediction = prediction[:, 0]
            if target_resized.ndim == 4:
                target_resized = target_resized[:, 0]
            if mask_resized.ndim == 4:
                mask_resized = mask_resized[:, 0]

            valid = (
                (mask_resized > 0.5)
                & torch.isfinite(prediction)
                & torch.isfinite(target_resized)
            )

            count = int(valid.sum().item())

            if count == 0:
                continue

            diff = prediction - target_resized

            total_abs += float(
                torch.abs(diff)[valid].sum().cpu()
            )
            total_sq += float(
                (diff * diff)[valid].sum().cpu()
            )

            valid_pixels += count
            samples += 1

            mae_now = total_abs / max(valid_pixels, 1)
            rmse_now = math.sqrt(
                total_sq / max(valid_pixels, 1)
            )

            print(
                f"Validation {samples:02d}/{requested_samples} | "
                f"MAE {mae_now:.5f} | "
                f"RMSE {rmse_now:.5f} | "
                f"Scene {scene} | "
                f"Scanned {scanned} | "
                f"Time {time.time() - start:.1f}s"
            )

        except (OSError, RuntimeError) as exc:
            print(
                f"[WARNING] Validation sample failed "
                f"(scene={scene}): "
                f"{type(exc).__name__}: {exc}"
            )
            continue

    if samples == 0 or valid_pixels == 0:
        print()
        print(
            "[WARNING] Validation produced zero usable samples."
        )
        print(
            "[INFO] Validation is marked UNAVAILABLE; "
            "MAE=0 is never fabricated."
        )

        return {
            "mae": float("inf"),
            "rmse": float("inf"),
            "valid_pixels": 0,
            "samples": 0,
            "scanned_pairs": int(scanned),
            "validation_mode": policy.mode,
            "validation_scenes": sorted(
                policy.validation_scenes
            ),
            "status": "unavailable",
        }

    mae = total_abs / valid_pixels
    rmse = math.sqrt(total_sq / valid_pixels)

    if samples < requested_samples:
        print()
        print(
            f"[WARNING] Validation stream ended after "
            f"{samples}/{requested_samples} samples."
        )
        print(
            "[OK] Using the real samples that were successfully obtained."
        )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "valid_pixels": int(valid_pixels),
        "samples": int(samples),
        "scanned_pairs": int(scanned),
        "validation_mode": policy.mode,
        "validation_scenes": sorted(
            policy.validation_scenes
        ),
        "status": "ok",
    }


# ============================================================================
# CHECKPOINTS
# ============================================================================

def atomic_torch_save(
    payload: Dict[str, Any],
    path: Path,
) -> None:
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        tmp,
    )

    # os.replace is atomic on the same filesystem on Windows.
    os.replace(
        tmp,
        path,
    )


def build_checkpoint_payload(
    model: nn.Module,
    optimizer,
    scaler,
    epoch: int,
    train_loss: float,
    train_samples: int,
    validation: Dict[str, Any],
    best_mae: float,
) -> Dict[str, Any]:
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    payload: Dict[str, Any] = {
        "stage": "S-EO",
        "dataset": "emasquil/shadow-eo",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": float(train_loss),
        "train_samples": int(train_samples),
        "validation": validation,
        "best_validation_mae": float(best_mae),
        "config": {
            "encoder": ENCODER,
            "features": FEATURES,
            "out_channels": OUT_CHANNELS,
            "patch_size": PATCH_SIZE,
            "model_size": MODEL_SIZE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUMULATION,
            "effective_batch_size": (
                BATCH_SIZE * GRAD_ACCUMULATION
            ),
            "full_parameter_finetuning": True,
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "amp": AMP_ENABLED,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "optimizer": type(optimizer).__name__,
            "streaming": True,
            "global_index": False,
            "full_corpus_download": False,
            "local_dsm_archive_cached": True,
            "local_dsm_scene_cache": str(DSM_SCENE_DIR),
            "max_validation_samples": DEFAULT_VAL_STEPS,
            "recoverable_remote_stream_errors": True,
            "grad_scaler_safe_step": True,
            "rgb_resources": RGB_PARTS,
            "dsm_resource": DSM_URL,
            "initial_checkpoint": str(VAIHINGEN_CHECKPOINT),
            "validation_mode": validation.get(
                "validation_mode",
                "unknown",
            ),
            "validation_scenes": validation.get(
                "validation_scenes",
                [],
            ),
        },
    }

    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()

    return payload


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scaler,
    epoch: int,
    train_loss: float,
    train_samples: int,
    validation: Dict[str, Any],
    best_mae: float,
) -> None:
    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        train_loss=train_loss,
        train_samples=train_samples,
        validation=validation,
        best_mae=best_mae,
    )

    atomic_torch_save(
        payload,
        path,
    )

    print()
    print(f"Checkpoint saved: {path}")


def load_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(
            HISTORY_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):
            return data

    except Exception as exc:
        print(
            f"[WARNING] Could not read training history: "
            f"{type(exc).__name__}: {exc}"
        )

    return []


def get_best_mae_from_history(
    history: List[Dict[str, Any]],
) -> float:
    values = []

    for item in history:
        try:
            value = item["validation"]["mae"]
            values.append(float(value))
        except Exception:
            continue

    return min(values) if values else float("inf")


def resume_from_latest(
    model: nn.Module,
    optimizer,
    scaler,
) -> Tuple[int, float, Optional[Dict[str, Any]]]:
    """
    Returns:
        start_epoch,
        best_mae,
        checkpoint_payload

    If seo_latest.pth exists:
        model + optimizer + scaler are resumed.
        Training continues from latest completed epoch + 1.

    If it does not exist:
        model remains initialized from Vaihingen.
    """

    if not LATEST_CHECKPOINT.exists():
        print()
        print("[INFO] No seo_latest.pth found.")
        print("[INFO] Starting S-EO from Stage-1 Vaihingen weights.")
        return 1, float("inf"), None

    print()
    print("=" * 78)
    print("RESUMING S-EO TRAINING")
    print("=" * 78)
    print(f"Latest checkpoint: {LATEST_CHECKPOINT}")

    payload = getattr(model, "_seo_resume_payload", None)
    if payload is None:
        payload = torch.load(
            LATEST_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        state = _extract_state_dict(payload)
        state = _clean_state_dict(state)

        missing, unexpected = model.load_state_dict(
            state,
            strict=False,
        )

        if missing or unexpected:
            raise RuntimeError(
                "seo_latest.pth is incompatible with the current model.\n"
                f"Missing keys: {len(missing)}\n"
                f"Unexpected keys: {len(unexpected)}"
            )
    else:
        print("[OK] Reusing already-loaded S-EO checkpoint metadata.")

    optimizer_loaded = False

    if isinstance(payload, dict):
        optimizer_state = payload.get(
            "optimizer_state_dict"
        )

        if isinstance(optimizer_state, dict):
            try:
                optimizer.load_state_dict(
                    optimizer_state
                )
                optimizer_loaded = True
            except Exception as exc:
                print()
                print(
                    "[WARNING] Existing optimizer state could not be "
                    "loaded into the current optimizer."
                )
                print(
                    f"          {type(exc).__name__}: {exc}"
                )
                print(
                    "[INFO] Model weights will still be resumed; "
                    "optimizer starts fresh."
                )

        scaler_state = payload.get(
            "scaler_state_dict"
        )

        if scaler is not None and isinstance(
            scaler_state,
            dict,
        ):
            try:
                scaler.load_state_dict(
                    scaler_state
                )
            except Exception as exc:
                print(
                    "[WARNING] AMP scaler state was not loaded: "
                    f"{type(exc).__name__}: {exc}"
                )

    completed_epoch = int(
        payload.get(
            "epoch",
            0,
        )
        if isinstance(payload, dict)
        else 0
    )

    # Release cached checkpoint metadata after optimizer/scaler restoration.
    if hasattr(model, "_seo_resume_payload"):
        try:
            delattr(model, "_seo_resume_payload")
        except Exception:
            pass

    checkpoint_best = float(
        payload.get(
            "best_validation_mae",
            float("inf"),
        )
        if isinstance(payload, dict)
        else float("inf")
    )

    history = load_history()
    history_best = get_best_mae_from_history(
        history
    )

    best_mae = min(
        checkpoint_best,
        history_best,
    )

    print(
        f"[OK] Resumed model from completed epoch "
        f"{completed_epoch}."
    )

    if optimizer_loaded:
        print("[OK] Optimizer state resumed.")
    else:
        print(
            "[WARNING] Optimizer state was unavailable; "
            "optimizer starts fresh."
        )

    if math.isfinite(best_mae):
        print(
            f"[OK] Best validation MAE so far: "
            f"{best_mae:.6f}"
        )
    else:
        print(
            "[INFO] No previous validation MAE was found."
        )

    print(
        f"[OK] Next epoch: {completed_epoch + 1}"
    )

    return (
        completed_epoch + 1,
        best_mae,
        payload if isinstance(payload, dict) else {},
    )


def save_interrupt_checkpoint(model, optimizer, scaler, epoch, step, train_loss, best_mae, policy):
    """Save an interrupted-epoch recovery checkpoint without advancing seo_latest."""
    path = OUTPUT_DIR / f"seo_interrupt_epoch{epoch}.pth"
    validation = {
        "status": "interrupted",
        "mae": float("inf"),
        "rmse": float("inf"),
        "valid_pixels": 0,
        "samples": 0,
        "validation_mode": policy.mode,
        "validation_scenes": sorted(policy.validation_scenes),
    }
    payload = build_checkpoint_payload(
        model=model, optimizer=optimizer, scaler=scaler,
        epoch=epoch, train_loss=train_loss, train_samples=step,
        validation=validation, best_mae=best_mae,
    )
    payload["interrupted"] = True
    payload["interrupted_step"] = int(step)
    atomic_torch_save(payload, path)
    print()
    print(f"[OK] Interrupted-epoch recovery checkpoint saved: {path}")
    return path


# ============================================================================
# FULL TRAINING
# ============================================================================

def run_training(
    requested_steps: int,
    requested_val_steps: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this full-parameter Stage-2 trainer."
        )

    dev = torch.device("cuda")

    print()
    print("=" * 78)
    print("S-EO DATA PREFLIGHT")
    print("=" * 78)

    # Critical performance fix: DSM is downloaded once and reused locally.
    dsm_scenes = ensure_local_dsm_cache()
    print(f"[OK] Local DSM cache available: {len(dsm_scenes)} scenes.")
    print("[OK] RGB remains streamed remotely.")

    policy, discovery_scanned = discover_validation_policy(
        scan_pairs=VALIDATION_DISCOVERY_PAIRS,
    )

    print()
    print("=" * 78)
    print("VALIDATION CONFIGURATION")
    print("=" * 78)
    print(f"Mode: {policy.describe()}")
    print(f"Discovery pairs scanned: {discovery_scanned}")

    if policy.mode == "scene":
        print(
            f"Held-out scenes: "
            f"{sorted(policy.validation_scenes)}"
        )
    else:
        print(
            "Validation uses deterministic RGB member keys."
        )

    # ------------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------------

    # If an S-EO checkpoint exists, load it directly as the model initializer.
    # This avoids first loading Vaihingen and then replacing it.
    initial_checkpoint = (
        LATEST_CHECKPOINT
        if LATEST_CHECKPOINT.exists()
        else VAIHINGEN_CHECKPOINT
    )
    model = create_model(initial_checkpoint=initial_checkpoint)

    enable_gradient_checkpointing(
        model
    )

    parameter_report(
        model
    )

    model.to(dev)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print_memory(
        "after model loading"
    )

    optimizer = create_optimizer(
        model
    )

    scaler = create_scaler()

    # ------------------------------------------------------------------------
    # RESUME
    # ------------------------------------------------------------------------

    start_epoch, best_mae, _ = resume_from_latest(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
    )

    history = load_history()

    if not math.isfinite(best_mae):
        best_mae = get_best_mae_from_history(
            history
        )

    print()
    print("=" * 78)
    print("STARTING S-EO FULL-PARAMETER FINE-TUNING")
    print("=" * 78)

    print()
    print("Training chain:")
    print("Depth Anything V2 Large")
    print("        ↓")
    print("GeoNRW")
    print("        ↓")
    print("Potsdam")
    print("        ↓")
    print("Vaihingen")
    print("        ↓")
    print("S-EO  ← CURRENT STAGE")

    print()
    print("[OK] No global S-EO index.")
    print("[OK] No complete corpus download.")
    print("[OK] Multipart RGB stream is concatenated correctly.")
    print("[OK] DSM stream.")
    print("[OK] Full-parameter tuning.")
    print("[OK] Automatic resume from seo_latest.pth.")
    print(
        f"[OK] Validation: {policy.describe()}"
    )
    print(
        f"[OK] Validation maximum: {requested_val_steps} samples."
    )

    if start_epoch > EPOCHS:
        print()
        print(
            f"[INFO] Latest checkpoint is already at epoch "
            f"{start_epoch - 1}, while configured EPOCHS={EPOCHS}."
        )
        print(
            "[INFO] Nothing more is required under the current epoch "
            "configuration."
        )
        return

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):
        print()
        print("=" * 78)
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 78)

        # --------------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------------

        train_iterator = make_pair_stream(
            max_pairs=0
        )

        try:
            (
                train_loss,
                train_samples,
                exhausted,
            ) = train_steps(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                iterator=train_iterator,
                policy=policy,
                dev=dev,
                steps=requested_steps,
                epoch=epoch,
            )
        finally:
            del train_iterator
            gc.collect()
            torch.cuda.empty_cache()

        print()
        print(
            f"Epoch {epoch} training loss: "
            f"{train_loss:.6f}"
        )
        print(
            f"Epoch {epoch} training samples: "
            f"{train_samples}"
        )

        if exhausted:
            print(
                "[INFO] This epoch used fewer samples than the "
                "maximum requested step count because the remote "
                "stream ended."
            )

        # --------------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------------

        print()
        print("Running short validation...")

        val_iterator = make_pair_stream(
            max_pairs=0
        )

        try:
            validation = validate(
                model=model,
                iterator=val_iterator,
                policy=policy,
                dev=dev,
                requested_samples=requested_val_steps,
            )
        finally:
            del val_iterator
            gc.collect()
            torch.cuda.empty_cache()

        print()
        print("=" * 78)
        print("VALIDATION RESULT")
        print("=" * 78)
        print(
            f"Validation MAE:      "
            f"{validation['mae']:.6f}"
        )
        print(
            f"Validation RMSE:     "
            f"{validation['rmse']:.6f}"
        )
        print(
            f"Valid pixels:        "
            f"{validation['valid_pixels']:,}"
        )
        print(
            f"Validation samples:  "
            f"{validation['samples']:,}"
        )
        print(
            f"Validation mode:     "
            f"{validation['validation_mode']}"
        )

        # --------------------------------------------------------------
        # LATEST — always save the completed epoch.
        #
        # This is true even when the remote stream ended early or validation
        # was unavailable. The checkpoint therefore remains resumable.
        # --------------------------------------------------------------

        save_checkpoint(
            path=LATEST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            train_loss=train_loss,
            train_samples=train_samples,
            validation=validation,
            best_mae=best_mae,
        )

        # --------------------------------------------------------------
        # BEST — only replace when a REAL validation metric improves.
        # --------------------------------------------------------------

        validation_mae = float(
            validation.get("mae", float("inf"))
        )

        validation_available = (
            validation.get("status") == "ok"
            and math.isfinite(validation_mae)
            and validation.get("samples", 0) > 0
            and validation.get("valid_pixels", 0) > 0
        )

        if validation_available and validation_mae < best_mae:
            best_mae = validation_mae

            save_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                train_loss=train_loss,
                train_samples=train_samples,
                validation=validation,
                best_mae=best_mae,
            )

            print()
            print("NEW BEST S-EO CHECKPOINT")

        elif not validation_available:
            print()
            print(
                "[INFO] Best checkpoint unchanged because validation "
                "was unavailable."
            )

        else:
            print()
            print(
                "Best checkpoint kept unchanged "
                f"(best MAE = {best_mae:.6f})."
            )

        # --------------------------------------------------------------
        # HISTORY
        # --------------------------------------------------------------

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_samples": int(train_samples),
                "validation": validation,
                "best_validation_mae": float(best_mae),
            }
        )

        HISTORY_FILE.write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )

        print_memory(
            f"end of epoch {epoch}"
        )

    print()
    print("=" * 78)
    print("S-EO FULL-PARAMETER FINE-TUNING COMPLETE")
    print("=" * 78)
    print()
    print(
        f"Best validation MAE: {best_mae:.6f}"
    )
    print()
    print("Best checkpoint:")
    print(f"  {BEST_CHECKPOINT}")
    print()
    print("Latest checkpoint:")
    print(f"  {LATEST_CHECKPOINT}")
    print()
    print("Training history:")
    print(f"  {HISTORY_FILE}")
    print_memory("final")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "ASTERRA S-EO robust streaming full-parameter trainer"
        )
    )

    parser.add_argument(
        "--stream-test",
        action="store_true",
        help="Test RGB ↔ DSM streaming pairs only.",
    )

    parser.add_argument(
        "--validation-stream-test",
        action="store_true",
        help=(
            "Test validation streaming without constructing a dataset index."
        ),
    )

    parser.add_argument(
        "--gpu-smoke-test",
        action="store_true",
        help="Run one real GPU forward/backward/optimizer step.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=(
            "Maximum training samples/iterations per epoch. "
            f"Default: {DEFAULT_STEPS}"
        ),
    )

    parser.add_argument(
        "--val-steps",
        type=int,
        default=DEFAULT_VAL_STEPS,
        help=(
            "Maximum validation samples per epoch. "
            f"Default: {DEFAULT_VAL_STEPS}"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    seed_everything(SEED)
    print_gpu_info()

    if args.stream_test:
        raise SystemExit(
            0 if stream_test(10) else 1
        )

    if args.validation_stream_test:
        raise SystemExit(
            0
            if validation_stream_test(
                target=5,
                scan_pairs=VALIDATION_DISCOVERY_PAIRS,
            )
            else 1
        )

    if args.gpu_smoke_test:
        raise SystemExit(
            0 if gpu_smoke_test() else 1
        )

    if args.steps <= 0:
        raise ValueError(
            "--steps must be greater than zero."
        )

    if args.val_steps <= 0:
        raise ValueError(
            "--val-steps must be greater than zero."
        )

    run_training(
        requested_steps=args.steps,
        requested_val_steps=min(args.val_steps, DEFAULT_VAL_STEPS),
    )


if __name__ == "__main__":
    main()