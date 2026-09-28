"""Phase 9: segmentation package interface."""
from .class_map import (
    BARE_GROUND,
    BUILDING,
    COLORS,
    INFRASTRUCTURE,
    NAMES,
    OTHER,
    ROAD,
    VEGETATION,
    WATER,
    ade_labels_to_ours,
    valid_mask_ids,
)

__all__ = [
    "OTHER", "BUILDING", "ROAD", "VEGETATION", "WATER",
    "BARE_GROUND", "INFRASTRUCTURE", "NAMES", "COLORS",
    "ade_labels_to_ours", "valid_mask_ids",
]
