"""Phase 9: independent optional segmentation subsystem.

Public interface: SegmentationModel.predict(image) -> SegmentationResult.
SegFormer checkpoint is configurable (DW_SEGMENTATION_MODEL); architecture and
class mapping are deliberately separated (see class_map.ade_labels_to_ours).

If the model is unavailable the engine reports the reason and emits a clearly
labelled deterministic RGB fallback. The result is semantic evidence only;
height geometry still has to pass the DSM/ground gates downstream.
"""
import threading

import numpy as np
from PIL import Image

try:
    from ..config import settings
except ImportError:  # pragma: no cover - direct module execution fallback
    from backend.config import settings


SEG_CONFIDENCE = settings.segmentation_confidence
SEG_DEVICE = settings.segmentation_device
SEG_ENABLED = settings.segmentation_enabled
SEG_MAX_SIDE = settings.segmentation_max_side
SEG_MODEL_ID = settings.segmentation_model
SEG_LOCAL_ONLY = getattr(settings, "segmentation_local_only", True)
from .class_map import OTHER, ade_labels_to_ours


class SegmentationResult:
    def __init__(self, mask, confidence, classes_present, mean_confidence, model_id):
        self.mask = mask  # uint8 HxW, OUR class ids, input resolution
        self.confidence = confidence  # float32 HxW max-softmax
        self.classes_present = classes_present  # sorted list of our ids
        self.mean_confidence = mean_confidence  # over non-OTHER pixels
        self.model_id = model_id


class SegmentationModel:
    def predict(self, image):
        raise NotImplementedError


def _read_prep_config(model_id: str) -> dict:
    """Preprocessing params without torchvision: read the checkpoint's
    preprocessor_config.json from the hub cache (or defaults)."""
    prep = {"size": 512, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    try:
        if SEG_LOCAL_ONLY:
            return prep
        from huggingface_hub import snapshot_download

        d = snapshot_download(model_id, allow_patterns=["preprocessor_config.json"])
        import json as _json
        from pathlib import Path as _P

        cfg = _json.loads((_P(d) / "preprocessor_config.json").read_text())
        size = cfg.get("size", 512)
        if isinstance(size, dict):
            size = size.get("shortest_edge", size.get("height", 512))
        prep["size"] = int(size)
        if isinstance(cfg.get("image_mean"), list):
            prep["mean"] = [float(v) for v in cfg["image_mean"]]
        if isinstance(cfg.get("image_std"), list):
            prep["std"] = [float(v) for v in cfg["image_std"]]
    except Exception as exc:
        print(f"[segmentation] prep config fallback (ImageNet defaults): {exc}")
    return prep


class SegformerModel(SegmentationModel):
    def __init__(self, model_id=SEG_MODEL_ID, device=SEG_DEVICE):
        self._lock = threading.Lock()
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
        self.ade_lut = None
        self.tried = False
        self.available = False
        self.reason = None

    def _load(self) -> bool:
        with self._lock:
            if self.tried:
                return self.available
            self.tried = True
            try:
                from ..config import apply_torch_threads

                apply_torch_threads()
                import torch
                from transformers import (
                    SegformerForSemanticSegmentation,
                    SegformerImageProcessor,
                )

                torch_device = "cuda" if (
                    self.device in ("auto", "cuda") and torch.cuda.is_available()
                ) else "cpu"
                self.torch_device = torch_device
                # The HF image processor requires torchvision (not installed);
                # preprocess manually with PIL/numpy instead (rescale +
                # ImageNet normalize, params read from the checkpoint config).
                try:
                    from transformers import SegformerImageProcessor

                    self.processor = SegformerImageProcessor.from_pretrained(
                        self.model_id, local_files_only=SEG_LOCAL_ONLY
                    )
                    self._prep = None
                except Exception:
                    self.processor = None
                    self._prep = _read_prep_config(self.model_id)
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    self.model_id, local_files_only=SEG_LOCAL_ONLY
                )
                self.model.to(torch_device)
                self.model.eval()
                id2label = getattr(self.model.config, "id2label", {}) or {}
                self.ade_lut = ade_labels_to_ours(
                    {int(k): v for k, v in dict(id2label).items()}
                )
                self.available = True
            except Exception as exc:
                self.reason = str(exc)[:300]
                self.processor = None
                self.model = None
                self.available = False
                return False
            return True

    def status(self) -> dict:
        dependency_available = False
        dependency_reason = None
        try:
            import transformers  # noqa: F401
            dependency_available = True
        except Exception as exc:
            dependency_reason = str(exc)[:220]
        return {
            "enabled": SEG_ENABLED,
            "available": self.available,
            "dependency_available": dependency_available,
            "model_configured": bool(self.model_id),
            "local_only": SEG_LOCAL_ONLY,
            "model": self.model_id if self.available else None,
            "device": getattr(self, "torch_device", None) if self.available else None,
            "reason": self.reason or dependency_reason,
        }

    def _downscale(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        scale = min(1.0, SEG_MAX_SIDE / max(h, w))
        if scale >= 1.0:
            return rgb
        img = Image.fromarray(rgb).resize(
            (max(2, round(w * scale)), max(2, round(h * scale))), Image.BILINEAR
        )
        return np.asarray(img)

    def _manual_prep(self, small: np.ndarray):
        """torchvision-free preprocessing: short-edge resize + rescale +
        normalize, mirroring SegformerImageProcessor defaults."""
        import torch

        prep = self._prep or {}
        edge = int(prep.get("size", 512))
        h, w = small.shape[:2]
        s = edge / min(h, w)
        img = Image.fromarray(small).resize(
            (max(2, round(w * s)), max(2, round(h * s))), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        mean = np.array(prep.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(prep.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr.transpose(2, 0, 1)[None])

    def predict(self, rgb: np.ndarray) -> SegmentationResult | None:
        """Run segmentation. Returns None when unavailable (never raises)."""
        if not SEG_ENABLED:
            self.reason = self.reason or "disabled by DW_SEGMENTATION_ENABLED=false"
            return None
        if not self._load() or self.model is None:
            return None
        try:
            import torch
            import torch.nn.functional as F

            small = self._downscale(rgb)
            if self.processor is not None:
                inputs = self.processor(images=Image.fromarray(small), return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.torch_device)
            else:
                pixel_values = self._manual_prep(small).to(self.torch_device)
                inputs = None
            with torch.no_grad():
                logits = self.model(pixel_values=pixel_values).logits  # 1,C,hs,ws
            up = F.interpolate(
                logits, size=small.shape[:2], mode="bilinear", align_corners=False
            )
            prob = F.softmax(up, dim=1)[0]
            top = prob.max(dim=0)
            conf = top.values.cpu().numpy().astype(np.float32)
            ade = top.indices.cpu().numpy()
            lut = self.ade_lut
            ours = lut[np.clip(ade, 0, len(lut) - 1)].astype(np.uint8)
            # Confidence gate: uncertain pixels are OTHER, never forced.
            ours = np.where(conf >= SEG_CONFIDENCE, ours, OTHER).astype(np.uint8)
            if (ours.shape[0], ours.shape[1]) != rgb.shape[:2]:
                ours = np.asarray(
                    Image.fromarray(ours).resize(
                        (rgb.shape[1], rgb.shape[0]), Image.NEAREST
                    )
                ).astype(np.uint8)
                conf = np.asarray(
                    Image.fromarray(conf, mode="F").resize(
                        (rgb.shape[1], rgb.shape[0]), Image.BILINEAR
                    )
                ).astype(np.float32)
            present = sorted(int(c) for c in np.unique(ours).tolist())
            fg = ours != OTHER
            mean_conf = float(conf[fg].mean()) if fg.any() else 0.0
            del inputs, pixel_values, logits, up, prob
            return SegmentationResult(ours, conf, present, mean_conf, self.model_id)
        except Exception as exc:
            self.reason = f"inference failed: {str(exc)[:200]}"
            return None


ENGINE = SegformerModel()


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """Small dependency-free box filter used by the emergency fallback."""
    radius = max(1, int(radius))
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    ) / float(size * size)


def _heuristic_segmentation(rgb: np.ndarray) -> SegmentationResult:
    """Produce useful semantic evidence without a downloaded ML checkpoint.

    This is intentionally conservative and is never labelled as model output.
    The downstream geometry gates it with local height evidence, so a bright
    road or courtyard cannot automatically become a tall building.
    """
    from .class_map import BUILDING, ROAD, VEGETATION, WATER, BARE_GROUND, INFRASTRUCTURE

    image = np.asarray(rgb, dtype=np.float32) / 255.0
    image = np.clip(image, 0.0, 1.0)
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    high = np.maximum.reduce((r, g, b))
    low = np.minimum.reduce((r, g, b))
    saturation = (high - low) / np.maximum(high, 1e-3)

    # Local texture separates uniform roof/road surfaces from foliage.  It is
    # only evidence; the building extractor still requires positive relief.
    radius = max(2, min(image.shape[:2]) // 96)
    local = _box_mean(luminance, radius)
    local_sq = _box_mean(luminance * luminance, radius)
    texture = np.sqrt(np.maximum(local_sq - local * local, 0.0))

    green_index = g - (r + b) * 0.5
    blue_index = b - (r + g) * 0.5
    blue_green_index = (b + g) * 0.5 - r

    # The old fallback treated almost every non-green, non-blue pixel as a
    # roof. That is especially destructive for brown soil and pale courtyards.
    # Use mutually exclusive evidence scores instead. Dark ponds are often
    # teal/green rather than strongly blue, so water includes a low-texture
    # blue-green branch as well as the conventional blue branch.
    water_score = (
        (blue_index > 0.012) & (luminance < 0.78) & (texture < 0.20)
    ) | (
        (blue_green_index > 0.018) & (luminance < 0.58) & (texture < 0.16)
    )
    water_score &= (g >= r * 0.98) & (b >= r * 0.98)

    vegetation_score = (
        (green_index > 0.025)
        & (g > r * 1.03)
        & (g >= b * 0.98)
        & (luminance > 0.08)
    ) & ~water_score

    neutral_roof = (saturation < 0.42) & (luminance > 0.32) & (luminance < 0.92)
    warm_roof = (r > b * 1.10) & (g > b * 1.04) & (saturation < 0.56) & (luminance > 0.36)
    roof_like = (
        ~water_score
        & ~vegetation_score
        & (neutral_roof | warm_roof)
        & (texture < 0.16)
        & (local > 0.24)
    )
    road = (
        ~water_score
        & ~vegetation_score
        & ~roof_like
        & (saturation < 0.24)
        & (luminance < 0.52)
        & (texture < 0.19)
    )
    bare = (
        ~water_score
        & ~vegetation_score
        & ~roof_like
        & ~road
        & (r >= b * 1.04)
        & (luminance > 0.16)
    )
    infrastructure = (
        ~water_score
        & ~vegetation_score
        & ~roof_like
        & ~road
        & ~bare
        & (luminance > 0.62)
    )

    # Remove isolated salt-and-pepper pixels while retaining large roofs,
    # ponds, and tree-covered regions. scipy is already a backend dependency;
    # the fallback remains functional if it is unavailable.
    def clean(binary, minimum):
        try:
            from scipy import ndimage

            binary = ndimage.binary_closing(binary, iterations=1)
            labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
            if count:
                sizes = np.bincount(labels.ravel())
                binary = binary & (sizes[labels] >= int(minimum))
            return binary
        except Exception:
            return binary

    pixels = max(1, luminance.size)
    water_score = clean(water_score, max(12, pixels // 18000))
    vegetation_score = clean(vegetation_score, max(20, pixels // 12000))
    roof_like = clean(roof_like, max(12, pixels // 16000))
    road = clean(road, max(8, pixels // 24000))

    mask = np.zeros(luminance.shape, dtype=np.uint8)
    confidence = np.full(luminance.shape, 0.35, dtype=np.float32)
    mask[bare] = BARE_GROUND
    mask[road] = ROAD
    mask[infrastructure] = INFRASTRUCTURE
    mask[vegetation_score] = VEGETATION
    mask[water_score] = WATER
    mask[roof_like] = BUILDING

    vegetation_confidence = np.clip(0.60 + green_index * 2.4 + texture * 0.35, 0.55, 0.92)
    water_confidence = np.clip(0.60 + np.maximum(blue_index, blue_green_index) * 2.8, 0.55, 0.90)
    roof_confidence = np.clip(0.58 + (0.16 - texture) * 1.2, 0.55, 0.86)
    confidence[vegetation_score] = vegetation_confidence[vegetation_score]
    confidence[water_score] = water_confidence[water_score]
    confidence[roof_like] = roof_confidence[roof_like]
    present = sorted(int(value) for value in np.unique(mask).tolist())
    foreground = mask != OTHER
    return SegmentationResult(
        mask=mask,
        confidence=confidence,
        classes_present=present,
        mean_confidence=float(confidence[foreground].mean()) if foreground.any() else 0.0,
        model_id="heuristic-rgb-v1",
    )


def segment_rgb(rgb: np.ndarray, allow_model: bool = True):
    """Convenience: (result | None, info dict for result.json)."""
    res = ENGINE.predict(rgb) if allow_model else None
    if res is None:
        if not SEG_ENABLED:
            return None, {
                "enabled": False,
                "available": False,
                "fallback": True,
                "reason": getattr(ENGINE, "reason", None) or "segmentation disabled",
            }
        # Keep semantic evidence available even when transformers/model weights
        # are absent. This path is bounded, deterministic, and lets the mesh
        # isolate roofs, vegetation, and water instead of turning all of them
        # into one noisy height field.
        heuristic = _heuristic_segmentation(rgb)
        return heuristic, {
            "enabled": True,
            "available": False,
            "fallback": True,
            "reason": getattr(ENGINE, "reason", None) or "model unavailable",
            "model": heuristic.model_id,
            "classes_present": [int(c) for c in heuristic.classes_present],
            "mean_confidence": round(heuristic.mean_confidence, 3),
        }
    return res, {
        "enabled": True,
        "available": True,
        "fallback": False,
        "model": res.model_id,
        "classes_present": [int(c) for c in res.classes_present],
        "mean_confidence": round(res.mean_confidence, 3),
    }
