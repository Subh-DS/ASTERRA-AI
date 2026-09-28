"""Adapter around the frozen ASTERRA tiled inference implementation."""

import importlib
from pathlib import Path
import sys
import threading


class InferenceService:
    def __init__(self):
        self._lock = threading.Lock()
        self._module = None
        self._model = None
        self._device = None

    def _load_module(self, settings):
        if str(settings.project_root) not in sys.path:
            sys.path.insert(0, str(settings.project_root))
        if str(settings.depth_anything_root) not in sys.path:
            sys.path.insert(0, str(settings.depth_anything_root))
        if self._module is None:
            self._module = importlib.import_module("asterra_tiled_inference")
        return self._module

    def health(self, settings):
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return {
                "device": device,
                "model": "ASTERRA Stage 5",
                "model_loaded": self._model is not None,
                "checkpoint_exists": settings.model_path.exists(),
                "depth_backend_available": settings.depth_anything_root.exists(),
            }
        except Exception as exc:
            return {"device": "unavailable", "model": "ASTERRA Stage 5", "model_loaded": False, "error": str(exc)}

    def _ensure_model(self, settings):
        module = self._load_module(settings)
        with self._lock:
            if self._model is None:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = module.build_model(settings.model_path, self._device)
        return module, self._device, self._model

    def run(self, settings, input_path, output_npy, output_tif, progress_callback=None):
        module, device, model = self._ensure_model(settings)
        return module.run_inference(
            input_path=input_path,
            output_npy=output_npy,
            output_tif=output_tif,
            checkpoint_path=settings.model_path,
            device=device,
            model=model,
            progress_callback=progress_callback,
        )


inference_service = InferenceService()
