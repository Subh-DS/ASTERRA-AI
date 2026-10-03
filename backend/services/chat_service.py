"""Geospatial-only chat service for the ASTERRA field assistant.

This service provides strictly geospatial responses. It uses a comprehensive
local knowledge base covering remote sensing, photogrammetry, GIS, elevation
models, and the ASTERRA pipeline. For open-ended geospatial questions, it can
optionally use the Hugging Face Inference API (free tier) with a geospatial
system prompt. Falls back to the local knowledge base if the API is unavailable.
"""

import os
from pathlib import Path

# Load .env file from project root (two levels up from backend/services/)
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)




GEOSPATIAL_SYSTEM_PROMPT = """You are the ASTERRA Field Assistant, a geospatial AI copilot.
You ONLY answer questions related to geospatial topics including:
- Remote sensing and satellite imagery
- Photogrammetry and 3D reconstruction
- GIS (Geographic Information Systems)
- Elevation models (DSM, DTM, DEM, nDSM)
- Terrain analysis and visualization
- Georeferencing and coordinate systems (CRS, projections)
- Ground control points (GCPs) and calibration
- LiDAR and point clouds
- SAR (Synthetic Aperture Radar)
- Hyperspectral/multispectral imagery
- Cartography and map-making
- Geospatial data formats (GeoTIFF, GeoJSON, LAS, etc.)
- The ASTERRA pipeline and its outputs
- Building extraction and land cover classification
- Flood modeling and hazard assessment
- Urban 3D modeling

If a question is NOT related to geospatial topics, politely decline and redirect
the user to ask geospatial questions. Keep responses concise, technical, and
practical. Use markdown formatting when helpful."""

# Comprehensive geospatial knowledge base for local responses
GEOSPATIAL_KB = {
    "dsm": {
        "keywords": ["dsm", "digital surface model", "surface model"],
        "response": (
            "A **Digital Surface Model (DSM)** represents the Earth's surface including all objects on it "
            "(buildings, trees, infrastructure). It is typically derived from LiDAR, photogrammetry, or "
            "monocular depth estimation. In ASTERRA, the DSM is produced by calibrating relative depth "
            "predictions against a reference DEM (e.g., SRTM Terrarium) using RANSAC affine fitting, "
            "yielding metric elevation values in meters."
        ),
    },
    "dtm": {
        "keywords": ["dtm", "digital terrain model", "terrain model", "bare earth"],
        "response": (
            "A **Digital Terrain Model (DTM)** represents the bare ground surface with all objects "
            "(buildings, vegetation) removed. It is useful for hydrological analysis, slope mapping, and "
            "terrain characterization. ASTERRA focuses on DSM production; DTM generation typically "
            "requires ground-point classification from LiDAR point clouds."
        ),
    },
    "dem": {
        "keywords": ["dem", "digital elevation model", "elevation model", "elevation data"],
        "response": (
            "A **Digital Elevation Model (DEM)** is a generic term for raster elevation data. It can "
            "represent either the bare ground (DTM) or surface with objects (DSM). Common sources include "
            "SRTM (30m global), Copernicus GLO-30 (30m global), ALOS World 3D (30m), and LiDAR "
            "(sub-meter to meter resolution). ASTERRA uses DEM references for metric calibration of "
            "monocular depth predictions."
        ),
    },
    "ndsm": {
        "keywords": ["ndsm", "normalized dsm", "normalized surface", "height above ground", "canopy height"],
        "response": (
            "A **Normalized DSM (nDSM)** represents height above ground level, computed as DSM minus DTM. "
            "It is essential for building height estimation, vegetation analysis, and urban modeling. "
            "In ASTERRA, the monocular depth model produces a relative nDSM that is then calibrated "
            "to metric units."
        ),
    },
    "georeference": {
        "keywords": ["georeferenc", "crs", "coordinate reference", "projection", "epsg", "datum", "wgs84", "utm"],
        "response": (
            "**Georeferencing** ties raster pixels to real-world coordinates using a Coordinate Reference "
            "System (CRS). Common CRS include WGS84 (EPSG:4326, geographic) and UTM zones (EPSG:326xx/327xx, "
            "projected in meters). GeoTIFF files embed georeferencing via a GeoTransform (affine mapping "
            "from pixel to world coordinates). ASTERRA requires georeferenced inputs for metric DSM "
            "production; ungeoreferenced images produce relative surface models."
        ),
    },
    "gcp": {
        "keywords": ["gcp", "ground control", "control point", "ground truth point"],
        "response": (
            "**Ground Control Points (GCPs)** are known geographic coordinates tied to image pixels. "
            "They enable metric calibration of relative depth predictions. ASTERRA accepts GCPs as "
            "(lat, lon, elevation) tuples and fits an affine transform Z = a·d + b through them using "
            "RANSAC. At least 3 valid GCPs are required for metric calibration without a reference DEM."
        ),
    },
    "lidar": {
        "keywords": ["lidar", "las", "point cloud", "laser scanning"],
        "response": (
            "**LiDAR** (Light Detection and Ranging) uses laser pulses to measure distances, producing "
            "dense 3D point clouds. LAS/LAZ are standard point cloud formats. LiDAR is the gold standard "
            "for DSM/DTM generation, achieving sub-meter vertical accuracy. ASTERRA uses LiDAR-derived "
            "DEMs as reference surfaces for calibration validation."
        ),
    },
    "photogrammetry": {
        "keywords": ["photogrammetry", "sfm", "structure from motion", "mvs", "multi-view stereo"],
        "response": (
            "**Photogrammetry** extracts 3D geometry from overlapping 2D images. Structure from Motion "
            "(SfM) estimates camera poses and sparse point clouds; Multi-View Stereo (MVS) densifies them. "
            "ASTERRA uses monocular depth estimation (Depth Anything V2) as a lightweight alternative, "
            "producing relative height from single images, then calibrates to metric scale."
        ),
    },
    "sar": {
        "keywords": ["sar", "radar", "synthetic aperture", "interferometry", "insar"],
        "response": (
            "**SAR** (Synthetic Aperture Radar) uses microwave backscatter for all-weather, day-night "
            "imaging. InSAR (Interferometric SAR) measures surface deformation and elevation with "
            "centimeter-level precision. SAR is complementary to optical imagery for flood monitoring, "
            "subsidence detection, and DSM generation in cloudy regions."
        ),
    },
    "multispectral": {
        "keywords": ["multispectral", "hyperspectral", "spectral", "band", "ndvi", "vegetation index"],
        "response": (
            "**Multispectral** imagery captures data in discrete wavelength bands (e.g., Sentinel-2: 13 bands "
            "from 443nm to 2190nm). **Hyperspectral** sensors capture hundreds of narrow contiguous bands. "
            "Spectral indices like NDVI (Normalized Difference Vegetation Index) quantify vegetation health. "
            "ASTERRA primarily uses RGB imagery but can leverage multispectral data for land cover "
            "classification and environmental context."
        ),
    },
    "geotiff": {
        "keywords": ["geotiff", "tif", "tiff", "raster format", "geojson", "shapefile", "file format"],
        "response": (
            "**GeoTIFF** is a TIFF variant with embedded georeferencing metadata (CRS, transform, extent). "
            "It is the standard format for raster geospatial data. **GeoJSON** encodes vector features "
            "(points, lines, polygons) in JSON. **Shapefiles** (.shp) are a legacy vector format. ASTERRA "
            "accepts GeoTIFF, PNG, and JPG inputs and exports GeoTIFF DSMs, GeoJSON building footprints, "
            "and GLB 3D meshes."
        ),
    },
    "calibration": {
        "keywords": ["calibrat", "ransac", "affine", "scale fit", "vertical offset"],
        "response": (
            "**Calibration** in ASTERRA converts relative depth predictions to metric elevation. The "
            "pipeline uses RANSAC-based affine fitting (Z = a·d + b) against a reference DEM or GCPs. "
            "RANSAC robustly handles outliers by iteratively fitting on inlier subsets. The calibration "
            "metadata records the fit parameters, RMSE, and data source for traceability."
        ),
    },
    "pipeline": {
        "keywords": ["pipeline", "how does", "what does this", "under the hood", "works", "stages"],
        "response": (
            "The **ASTERRA pipeline** has four stages:\n"
            "1. **INGEST** — Parse image, detect georeferencing, run semantic segmentation\n"
            "2. **DEPTH** — Monocular depth estimation (Depth Anything V2 Large) over overlapping tiles\n"
            "3. **CALIBRATE** — RANSAC affine fit to reference DEM or GCPs for metric elevation\n"
            "4. **MESH** — Extrude calibrated heights into a textured, flyable 3D surface (GLB)\n\n"
            "Each stage reports progress and can be monitored in real-time."
        ),
    },
    "building": {
        "keywords": ["building", "footprint", "structure", "urban", "3d building"],
        "response": (
            "**Building extraction** in ASTERRA uses OpenStreetMap building footprints enriched with "
            "height tags (height, building:levels) or metric DSM samples. Buildings are extruded from "
            "the terrain surface and exported as part of the GLB mesh. The semantic segmentation "
            "model (SegFormer) provides additional building candidate detection from imagery."
        ),
    },
    "flood": {
        "keywords": ["flood", "inundation", "water level", "hazard", "storm surge"],
        "response": (
            "**Flood modeling** in ASTERRA uses the calibrated DSM to simulate water inundation. "
            "Given a water level (e.g., storm surge + sea level rise), the system identifies areas "
            "below that elevation and visualizes them as flood layers. The hazard module supports "
            "coastal inundation, storm scenarios, and evacuation planning."
        ),
    },
    "slope": {
        "keywords": ["slope", "aspect", "gradient", "steepness", "terrain analysis"],
        "response": (
            "**Slope** measures terrain steepness (degrees or percent rise), computed from elevation "
            "differences between neighboring cells. **Aspect** indicates the downslope direction. "
            "ASTERRA visualizes slope as a color ramp (teal flats → amber → red steep faces) and "
            "provides isoline overlays for contour-based terrain reading."
        ),
    },
    "contour": {
        "keywords": ["contour", "isoline", "elevation line", "topographic"],
        "response": (
            "**Contours** (isolines) connect points of equal elevation on a terrain surface. They "
            "provide a 2D topographic representation of 3D terrain. ASTERRA overlays real-data "
            "isolines on the 3D mesh for quantitative terrain reading, with configurable intervals."
        ),
    },
    "resolution": {
        "keywords": ["resolution", "gsd", "ground sample", "pixel size", "spatial resolution"],
        "response": (
            "**Ground Sample Distance (GSD)** is the ground distance represented by one pixel. "
            "It determines the level of detail recoverable from imagery. Sentinel-2 has ~10m GSD; "
            "high-resolution commercial satellites achieve 0.3–0.5m; LiDAR can reach sub-meter. "
            "ASTERRA warns when source GSD exceeds 5m/pixel (terrain-scale only) or 10m/pixel "
            "(building-level measurements disabled)."
        ),
    },
    "accuracy": {
        "keywords": ["accuracy", "rmse", "mae", "error", "precision", "validation"],
        "response": (
            "**Accuracy assessment** compares ASTERRA DSM outputs against reference data (e.g., LiDAR). "
            "Key metrics: **RMSE** (Root Mean Square Error), **MAE** (Mean Absolute Error), and "
            "**Pearson r** (correlation). ASTERRA reports these after validation and flags results "
            "above 6 m RMSE as having soft detail on fine structures."
        ),
    },
    "segmentation": {
        "keywords": ["segmentation", "land cover", "classification", "semantic", "mask"],
        "response": (
            "**Semantic segmentation** classifies each pixel into land cover categories (building, "
            "vegetation, water, road, etc.). ASTERRA uses SegFormer (ADE20K fine-tuned) for this task, "
            "producing a color-coded mask and building candidate detections. The segmentation feeds "
            "into building extraction and environmental context layers."
        ),
    },
    "mesh": {
        "keywords": ["mesh", "glb", "3d model", "terrain mesh", "reconstruction"],
        "response": (
            "**3D mesh reconstruction** extrudes the calibrated DSM into a triangulated surface with "
            "the source imagery as texture. ASTERRA exports **GLB** (GL Transmission Format Binary) "
            "for web-based 3D viewing. The mesh supports vertical exaggeration, hillshade lighting, "
            "slope visualization, and flythrough navigation."
        ),
    },
    "srtm": {
        "keywords": ["srtm", "terrarium", "copernicus", "elevation source", "dem source"],
        "response": (
            "**SRTM** (Shuttle Radar Topography Mission) provides global 30m elevation data. "
            "**Copernicus GLO-30** is a modern alternative with similar coverage. ASTERRA uses "
            "SRTM-derived Terrarium DEM tiles as the default reference for metric calibration, "
            "automatically downloading and mosaicking the relevant extent."
        ),
    },
    "coordinate": {
        "keywords": ["coordinate", "latitude", "longitude", "lat", "lon", "easting", "northing"],
        "response": (
            "**Geographic coordinates** use latitude (−90° to +90°) and longitude (−180° to +180°) "
            "on the WGS84 ellipsoid. **Projected coordinates** (e.g., UTM) use easting/northing in "
            "meters. ASTERRA handles coordinate transformations between the input CRS and the "
            "metric CRS used for DSM output."
        ),
    },
    "hillshade": {
        "keywords": ["hillshade", "shading", "lighting", "shadow", "relief"],
        "response": (
            "**Hillshade** simulates terrain illumination from a virtual sun position, enhancing "
            "the perception of relief and topography. ASTERRA applies hillshade as a lighting mode "
            "on the 3D mesh, complementing the optical texture and slope color ramp."
        ),
    },
    "exaggeration": {
        "keywords": ["exaggerat", "z scale", "vertical scale", "height scale"],
        "response": (
            "**Vertical exaggeration** scales the Z-axis of the 3D mesh to make subtle terrain "
            "features more visible. ASTERRA supports exaggeration from 0.2× to 4×. Use "
            "`set exaggeration to 2.5` to apply a 2.5× vertical scale."
        ),
    },
    "flythrough": {
        "keywords": ["fly", "flythrough", "first person", "cockpit", "navigate", "camera"],
        "response": (
            "**Fly mode** provides first-person navigation through the 3D terrain. Click the viewport "
            "to capture the mouse, then use WASD to move, Q/E for altitude, and Shift to sprint. "
            "Orbit mode (default) allows rotating and zooming around the terrain from an aerial "
            "perspective."
        ),
    },
    "map": {
        "keywords": ["map", "aoi", "area of interest", "imagery", "sentinel", "esri", "basemap"],
        "response": (
            "**Map jobs** let you define an Area of Interest (AOI) and fetch satellite imagery "
            "automatically. ASTERRA supports Sentinel-2 L2A (10m, free) via Earth Search STAC, "
            "Esri World Imagery (high-resolution), and optional high-resolution STAC catalogs. "
            "The AOI is clipped, processed, and reconstructed into a 3D terrain model."
        ),
    },
    "export": {
        "keywords": ["export", "download", "save", "output", "artifact"],
        "response": (
            "**Exportable artifacts** from ASTERRA include:\n"
            "- `dsm.tif` — GeoTIFF elevation raster (metric or relative)\n"
            "- `model.glb` — 3D textured mesh (GLB format)\n"
            "- `buildings.geojson` — Building footprints with heights\n"
            "- `environment.geojson` — Roads, water, landcover, trees\n"
            "- `mask.png` — Semantic segmentation preview\n"
            "- `metadata.json` — Full job metadata and quality metrics"
        ),
    },
    "validation": {
        "keywords": ["validate", "reference", "compare", "check accuracy", "ground truth"],
        "response": (
            "**Validation** compares the ASTERRA DSM against a reference DEM (e.g., LiDAR). Upload "
            "a reference GeoTIFF via the Validation panel. ASTERRA computes RMSE, MAE, and Pearson "
            "correlation, reporting whether the result is within the 6 m tolerance threshold."
        ),
    },
    "osm": {
        "keywords": ["osm", "openstreetmap", "overpass", "building data", "footprint data"],
        "response": (
            "**OpenStreetMap (OSM)** provides free vector data for buildings, roads, water bodies, "
            "and land cover. ASTERRA queries the Overpass API to fetch OSM features within the AOI, "
            "enriching the 3D mesh with building footprints, road ribbons, and environmental context. "
            "Building heights use OSM `height` or `building:levels` tags when available."
        ),
    },
    "stac": {
        "keywords": ["stac", "spatio-temporal", "catalog", "earth search", "cog"],
        "response": (
            "**STAC** (SpatioTemporal Asset Catalog) is a standard for organizing geospatial assets. "
            "ASTERRA uses STAC APIs (e.g., Earth Search) to search and download Cloud-Optimized "
            "GeoTIFFs (COGs) from Sentinel-2 and other catalogs. COGs enable efficient partial reads "
            "of large raster datasets."
        ),
    },
    "cloud": {
        "keywords": ["cloud", "cloud cover", "cloud mask", "cloud detection"],
        "response": (
            "**Cloud cover** affects optical imagery quality. ASTERRA filters scenes by cloud cover "
            "percentage when searching STAC catalogs. Cloud masks (e.g., Sentinel-2 SCL band) identify "
            "cloudy pixels for exclusion or flagging. High cloud cover degrades depth estimation "
            "accuracy."
        ),
    },
    "tile": {
        "keywords": ["tile", "tiling", "overlap", "stitch", "patch"],
        "response": (
            "**Tiled inference** splits large images into overlapping patches for memory-efficient "
            "processing. ASTERRA uses a sliding-window approach with overlap blending to stitch "
            "predictions seamlessly. Tile size and overlap are configurable to balance quality and "
            "performance."
        ),
    },
    "depth_anything": {
        "keywords": ["depth anything", "monocular", "depth estimation", "depth model"],
        "response": (
            "**Depth Anything V2 Large** is the monocular depth backbone used by ASTERRA. It predicts "
            "relative depth from a single RGB image without camera parameters. The model is "
            "fine-tuned on urban scenes for building and terrain height estimation. Predictions are "
            "relative (not metric) and require calibration against a reference DEM or GCPs."
        ),
    },
    "segformer": {
        "keywords": ["segformer", "segmentation model", "nvidia", "ade20k"],
        "response":(
            "**SegFormer**" (NVIDIA, ADE20K fine-tuned) is the semantic segmentation model used by "
            "ASTERRA for land cover classification. It produces pixel-level masks for buildings, "
            "vegetation, water, roads, and other categories. The model runs on GPU when available, "
            "with a deterministic RGB fallback for CPU-only environments."
        ),
    },
    "ransac": {
        "keywords": ["ransac", "robust fit", "outlier", "inlier"],
        "response": (
            "**RANSAC** (Random Sample Consensus) is a robust fitting algorithm that iteratively "
            "estimates model parameters from random subsets, favoring inliers over outliers. "
            "ASTERRA uses RANSAC for affine calibration (Z = a·d + b) to resist mismatches between "
            "predicted depth and reference elevation."
        ),
    },
    "affine": {
        "keywords": ["affine", "transform", "scale factor", "linear fit"],
        "response": (
            An **affine transform** maps relative depth to metric elevation via Z = a·d + b, where "
            "`a` is the scale factor and `b` is the vertical offset. ASTERRA fits this transform "
            "using RANSAC against reference DEM samples or GCPs, recording the parameters in "
            "calibration metadata."
        ),
    },
    "relative": {
        "keywords": ["relative", "no crs", "ungeoreferenced", "without georeference"],
        "response": (
            "**Relative mode** produces a surface model without metric scale when the input lacks "
            "georeferencing and no GCPs are provided. The output preserves relative height "
            "relationships (taller vs. shorter features) but cannot report absolute elevations. "
            "Add GCPs or use a georeferenced image for metric output."
        ),
    },
    "metric": {
        "keywords": ["metric", "meters", "absolute elevation", "elevation in meters"],
        "response": (
            "**Metric mode** produces elevation values in meters above a vertical datum (typically "
            "WGS84 ellipsoid or EGM96 geoid). Metric calibration requires either a georeferenced "
            "input with a reference DEM or at least 3 GCPs. The output DSM is suitable for "
            "engineering, hydrological, and planning applications."
        ),
    },
    "geoid": {
        "keywords": ["geoid", "egm96", "datum", "vertical datum", "ellipsoid"],
        "response": (
            "A **geoid** is the equipotential surface of Earth's gravity field that approximates mean "
            "sea level. **EGM96** is a common geoid model. Elevations can be referenced to the "
            "ellipsoid (h) or the geoid (H = h − N, where N is geoid undulation). ASTERRA uses "
            "DEM data with consistent vertical datum for calibration."
        ),
    },
    "utm": {
        "keywords": ["utm", "universal transverse mercator", "zone", "easting", "northing"],
        "response": (
            "**UTM** (Universal Transverse Mercator) divides Earth into 60 zones (6° longitude each), "
            "using easting/northing coordinates in meters. UTM is a common projected CRS for "
            "regional mapping and engineering. ASTERRA reprojects inputs to a suitable metric CRS "
            "for DSM production."
        ),
    },
    "wgs84": {
        "keywords": ["wgs84", "epsg:4326", "geographic coordinate"],
        "response": (
            "**WGS84** (EPSG:4326) is the global geographic coordinate system used by GPS. Coordinates "
            "are latitude/longitude in degrees. ASTERRA accepts WGS84 coordinates for GCPs and AOIs, "
            "transforming them to projected CRS for metric processing."
        ),
    },
    "copernicus": {
        "keywords": ["copernicus", "glo-30", "global dem"],
        "response": (
            "**Copernicus GLO-30** is a global 30m DEM derived from ALOS World 3D and other sources, "
            "provided free by the Copernicus programme. It offers improved accuracy over SRTM in "
            "vegetated and mountainous areas. ASTERRA can use Copernicus DEM as a calibration "
            "reference."
        ),
    },
    "alos": {
        "keywords": ["alos", "world 3d", "aw3d30"],
        "response": (
            "**ALOS World 3D (AW3D30)** is a global 30m DEM derived from ALOS PRISM stereo imagery. "
            "It provides higher vertical accuracy than SRTM in many regions. ASTERRA can use AW3D30 "
            "as a reference DEM for metric calibration."
        ),
    },
    "sentinel": {
        "keywords": ["sentinel", "sentinel-2", "esa", "copernicus data"],
        "response": (
            "**Sentinel-2** is ESA's multispectral imaging mission with two satellites (S2A, S2B), "
            "providing 10m resolution imagery in 13 spectral bands with 5-day revisit. ASTERRA uses "
            "Sentinel-2 L2A (atmospherically corrected) as the default free imagery source for map jobs."
        ),
    },
    "esri": {
        "keywords": ["esri", "world imagery", "arcgis", "basemap"],
        "response": (
            "**Esri World Imagery** is a high-resolution basemap service combining commercial satellite "
            "and aerial imagery (0.3–1m resolution in many areas). ASTERRA offers it as a selectable "
            "imagery provider for map jobs, with attribution retained in job metadata."
        ),
    },
    "overpass": {
        "keywords": ["overpass", "osm query", "building query"],
        "response": (
            "The **Overpass API** is a read-only API for querying OpenStreetMap data. ASTERRA uses "
            "it to fetch building footprints, roads, water bodies, and land cover within the AOI. "
            "Queries are spatial (bounding box) and return GeoJSON features for 3D enrichment."
        ),
    },
    "glb": {
        "keywords": ["glb", "gltf", "3d format", "model format"],
        "response": (
            "**GLB** (GL Transmission Format Binary) is a compact 3D model format combining geometry, "
            "materials, and textures in a single file. It is widely supported by web viewers, game "
            "engines, and 3D tools. ASTERRA exports terrain meshes as GLB for browser-based flythrough."
        ),
    },
    "hazard": {
        "keywords": ["hazard", "risk", "disaster", "earthquake", "storm", "evacuation"],
        "response": (
            "**Hazard assessment** in ASTERRA includes coastal inundation (sea level rise, storm "
            "surge), flood modeling, and earthquake impact visualization. The hazard module "
            "simulates water levels, identifies affected areas, and supports evacuation planning "
            "with animated scenarios."
        ),
    },
    "tree": {
        "keywords": ["tree", "vegetation", "forest", "canopy", "green area"],
        "response": (
            "**Tree and vegetation data** in ASTERRA comes from OSM land cover tags and semantic "
            "segmentation. Trees are represented as low-poly context geometry in the 3D mesh, "
            "with heights labeled as approximate when source data lacks measurements."
        ),
    },
    "road": {
        "keywords": ["road", "street", "highway", "transportation"],
        "response": (
            "**Road data** from OSM is rendered as ribbons on the terrain surface and in the GLB "
            "mesh. Road classification (motorway, primary, secondary, etc.) determines ribbon "
            "width and styling. Roads provide spatial context for urban scenes."
        ),
    },
    "water": {
        "keywords": ["water", "river", "lake", "ocean", "coastal", "shoreline"],
        "response": (
            "**Water features** (rivers, lakes, oceans) are fetched from OSM and rendered as flat "
            "blue surfaces in the 3D mesh. Coastal areas are identified for flood modeling. "
            "Water bodies provide context for terrain analysis and hazard assessment."
        ),
    },
    "quality": {
        "keywords": ["quality", "tier", "performance", "fps", "rendering"],
        "response": (
            "**Rendering quality** in ASTERRA adapts to device capabilities. Three tiers control "
            "mesh resolution (512/288/160 segments), shadows, fog, and effects. The quality "
            "detection runs at startup and can be overridden in Settings → Rendering quality."
        ),
    },
    "sample": {
        "keywords": ["sample", "demo", "example", "test scene", "load sample"],
        "response": (
            "To load a sample scene, use the **Map** screen to define an AOI and fetch imagery, "
            "or upload a GeoTIFF on the **Upload** screen. Sample AOIs with good building detail "
            "include urban areas with high-resolution Esri imagery."
        ),
    },
    "help": {
        "keywords": ["help", "what can you", "commands", "how to use"],
        "response": (
            "I can help with:\n"
            "- **Geospatial concepts** — DSM, DEM, DTM, nDSM, CRS, GCPs, LiDAR, SAR, etc.\n"
            "- **Pipeline explanation** — How ASTERRA processes imagery into 3D terrain\n"
            "- **Viewer commands** — `set exaggeration to 2.5`, `switch to fly`, `toggle isolines`\n"
            "- **Data sources** — Sentinel-2, Esri, SRTM, Copernicus, OSM\n"
            "- **Accuracy** — RMSE, MAE, validation against reference DEMs\n"
            "- **Export** — GeoTIFF, GLB, GeoJSON formats\n\n"
            "Ask me anything geospatial!"
        ),
    },
}


def _score_entry(query_lower: str, entry: dict) -> int:
    """Score a knowledge base entry against the query."""
    score = 0
    for keyword in entry["keywords"]:
        if keyword in query_lower:
            score += len(keyword)  # longer keyword matches are more specific
    return score


def _local_reply(query: str) -> str | None:
    """Generate a response from the local geospatial knowledge base."""
    query_lower = query.lower().strip()

    # Score all entries and pick the best match
    best_score = 0
    best_response = None
    for entry in GEOSPATIAL_KB.values():
        score = _score_entry(query_lower, entry)
        if score > best_score:
            best_score = score
            best_response = entry["response"]

    return best_response


def _is_geospatial(query: str) -> bool:
    """Check if a query is geospatial-related."""
    query_lower = query.lower()
    geospatial_indicators = [
        "dsm", "dem", "dtm", "ndsm", "elevation", "terrain", "height",
        "geospatial", "gis", "remote sensing", "satellite", "imagery",
        "photogrammetry", "lidar", "sar", "radar", "multispectral",
        "hyperspectral", "geotiff", "geojson", "shapefile", "raster",
        "vector", "coordinate", "projection", "crs", "epsg", "wgs84",
        "utm", "latitude", "longitude", "lat", "lon", "easting", "northing",
        "georeferenc", "gcp", "ground control", "calibrat", "ransac",
        "affine", "scale", "slope", "aspect", "contour", "isoline",
        "hillshade", "mesh", "glb", "3d", "building", "footprint",
        "flood", "inundation", "water", "hazard", "storm", "earthquake",
        "sentinel", "esri", "srtm", "copernicus", "alos", "terrarium",
        "stac", "cog", "cloud", "tile", "overlap", "depth", "monocular",
        "segformer", "segmentation", "land cover", "classification",
        "osm", "openstreetmap", "overpass", "road", "tree", "vegetation",
        "forest", "canopy", "coastal", "shoreline", "river", "lake",
        "ocean", "map", "aoi", "area of interest", "basemap",
        "resolution", "gsd", "pixel size", "accuracy", "rmse", "mae",
        "validation", "reference", "export", "download", "pipeline",
        "ingest", "depth", "calibrate", "mesh", "reconstruction",
        "exaggerat", "z scale", "fly", "flythrough", "orbit", "camera",
        "viewer", "sample", "demo", "help", "command",
        "geoid", "egm96", "datum", "ellipsoid", "vertical datum",
        "relative", "metric", "meter", "absolute", "quality", "tier",
        "render", "fps", "performance", "export", "artifact",
    ]
    return any(indicator in query_lower for indicator in geospatial_indicators)


def _non_geospatial_response(query: str) -> str:
    """Generate a polite decline for non-geospatial questions."""
    return (
        "I'm the ASTERRA Field Assistant, and I focus exclusively on geospatial topics. "
        "I can help with remote sensing, elevation models (DSM/DEM/DTM), terrain analysis, "
        "photogrammetry, GIS, LiDAR, SAR, building extraction, flood modeling, and the ASTERRA pipeline.\n\n"
        "Try asking me about:\n"
        "- \"What is a DSM?\"\n"
        "- \"How does calibration work?\"\n"
        "- \"Explain the pipeline\"\n"
        "- \"What is the difference between DSM and DTM?\"\n"
        "- \"How do I set up GCPs?\""
    )


def _openrouter_reply(messages: list[dict]) -> str | None:
    """Try to get a response from the OpenRouter API (free tier).

    Returns None if the API is not configured or the request fails.
    """
    token = os.getenv("OPENROUTER_API_KEY")
    if not token:
        return None

    try:
        model = os.getenv("ASTERRA_CHAT_MODEL", "google/gemma-4-26b-a4b-it:free")

        # Build the conversation with the geospatial system prompt
        conversation = [{"role": "system", "content": GEOSPATIAL_SYSTEM_PROMPT}]
        for msg in messages[-10:]:  # Keep last 10 messages for context
            role = "user" if msg.get("role") == "user" else "assistant"
            conversation.append({"role": role, "content": msg.get("content", "")})

        import urllib.request
        import json

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps({
                "model": model,
                "messages": conversation,
                "max_tokens": 512,
                "temperature": 0.7,
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://asterra.local",
                "X-Title": "ASTERRA Field Assistant",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception:
        return None


class ChatService:
    """Geospatial-only chat service."""

    def __init__(self):
        self._conversation_history = []

    def reply(self, messages: list[dict]) -> str:
        """Generate a geospatial-only response to the latest user message.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str} dicts.

        Returns:
            A geospatial-focused response string.
        """
        if not messages:
            return "Hello! I'm the ASTERRA Field Assistant. Ask me anything about geospatial topics — elevation models, remote sensing, terrain analysis, and more."

        # Extract the latest user message
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return "I'm here to help with geospatial questions. What would you like to know?"

        latest_query = user_messages[-1].get("content", "").strip()
        if not latest_query:
            return "Could you rephrase that? I'm best at geospatial questions."

        # Check if the query is geospatial
        if not _is_geospatial(latest_query):
            return _non_geospatial_response(latest_query)

        # Try local knowledge base first (fast, no API calls)
        local_response = _local_reply(latest_query)
        if local_response:
            return local_response

        # Try OpenRouter API for open-ended geospatial questions
        llm_response = _openrouter_reply(messages)
        if llm_response:
            return llm_response

        # Fallback for geospatial queries not in the knowledge base
        return (
            f"That's a geospatial question about **{latest_query[:50]}**. "
            "While I don't have a specific answer in my knowledge base, I can help with:\n"
            "- Elevation models (DSM, DTM, DEM, nDSM)\n"
            "- Remote sensing and satellite imagery\n"
            "- Photogrammetry and 3D reconstruction\n"
            "- GIS concepts and coordinate systems\n"
            "- Terrain analysis and visualization\n"
            "- The ASTERRA pipeline\n\n"
            "Try rephrasing or ask about a related geospatial topic."
        )


# Singleton instance
chat_service = ChatService()
