"""Phase 9: centralized semantic class definitions.

Integer IDs appear ONLY here. Everything else imports these names.
ADE20K label names (any SegFormer ADE20K checkpoint) map onto our
remote-sensing classes by name, so the checkpoint stays swappable.
"""
import numpy as np

OTHER = 0
BUILDING = 1
ROAD = 2
VEGETATION = 3
WATER = 4
BARE_GROUND = 5
INFRASTRUCTURE = 6

NAMES = {
    OTHER: "other",
    BUILDING: "building",
    ROAD: "road",
    VEGETATION: "vegetation",
    WATER: "water",
    BARE_GROUND: "bare_ground",
    INFRASTRUCTURE: "infrastructure",
}

# Distinct mask colors (R, G, B) for mask.png visualization.
COLORS = {
    OTHER: (0, 0, 0),
    BUILDING: (230, 70, 60),
    ROAD: (90, 90, 95),
    VEGETATION: (60, 160, 70),
    WATER: (60, 120, 230),
    BARE_GROUND: (190, 170, 120),
    INFRASTRUCTURE: (200, 130, 40),
}

# ADE20K label-name fragments (lowercase, substring match) per our class.
# Deliberately narrow for BUILDING: better to miss a shed than to invent one.
_ADE = {
    BUILDING: ("building", "house", "skyscraper", "hovel", "pand"),
    ROAD: ("road", "sidewalk", "runway", "path", "wegdeel", "weg"),
    VEGETATION: ("tree", "grass", "plant", "palm", "flower", "field", "forest",
                 "woods", "vegetatie", "vegetation"),
    WATER: ("water", "sea", "river", "lake", "pool", "waterfall", "waterdeel"),
    BARE_GROUND: ("earth", "sand", "rock", "mountain", "hill", "desert",
                  "soil", "terrain"),
    INFRASTRUCTURE: (
        "bridge", "overbruggingsdeel", "brug", "tower", "fence", "railing",
        "wall", "column", "car", "truck", "bus", "pole", "traffic",
        "signboard", "streetlight",
    ),
}


def ade_labels_to_ours(id2label: dict) -> np.ndarray:
    """Build a [n_ade] lookup of our class ids from a checkpoint's id2label.

    Unknown labels → OTHER. Never raises: unmapped checkpoints still run,
    they just classify everything as OTHER (honest, and covered by tests).
    """
    names = [str(id2label.get(i, "")).lower() for i in range(len(id2label))]
    lut = np.zeros(len(names), dtype=np.uint8)
    for ours, fragments in _ADE.items():
        for i, name in enumerate(names):
            if lut[i] != OTHER:
                continue
            if any(frag in name for frag in fragments):
                lut[i] = ours
    return lut


def valid_mask_ids(mask: np.ndarray) -> bool:
    return bool(np.all((mask >= 0) & (mask <= INFRASTRUCTURE)))
