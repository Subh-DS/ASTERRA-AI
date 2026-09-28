from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DEMSource:
    name: str
    path: Optional[Path]
    description: str
    resolution_m: Optional[float]
    source_type: str


DEM_SOURCES = {
    "srtm": DEMSource(
        name="SRTM",
        path=None,
        description="Shuttle Radar Topography Mission DEM",
        resolution_m=None,
        source_type="external_dem",
    ),

    "aw3d30": DEMSource(
        name="AW3D30",
        path=None,
        description="JAXA ALOS World 3D 30m DEM",
        resolution_m=None,
        source_type="external_dem",
    ),

    "glo30": DEMSource(
        name="GLO-30",
        path=None,
        description="Copernicus Global Digital Elevation Model",
        resolution_m=None,
        source_type="external_dem",
    ),

    "custom": DEMSource(
        name="Custom DEM",
        path=None,
        description="User-provided terrain elevation raster",
        resolution_m=None,
        source_type="custom_dem",
    ),
}


def get_dem_source(name: str) -> DEMSource:

    key = name.lower()

    if key not in DEM_SOURCES:
        raise ValueError(
            f"Unknown DEM source: {name}. "
            f"Available sources: "
            f"{', '.join(DEM_SOURCES.keys())}"
        )

    return DEM_SOURCES[key]


def describe_dem_source(name: str):

    source = get_dem_source(name)

    print("=" * 70)
    print("ASTERRA DEM SOURCE")
    print("=" * 70)

    print("Name        :", source.name)
    print("Description :", source.description)
    print("Resolution  :", source.resolution_m)
    print("Type        :", source.source_type)

    if source.path:
        print("Path        :", source.path)

    print("=" * 70)