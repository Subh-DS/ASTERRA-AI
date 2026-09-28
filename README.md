ASTERRA AI — Production Stage 5

Production checkpoint:
models/asterra_stage5/urban3d/stage5_urban3d_best.pth

Purpose:
Single-view height / relative height estimation, followed by geospatial calibration
to produce metric elevation / DSM outputs.

Production decision:

Urban3D Stage 5 is the locked safe checkpoint.

SpaceNet MVS direct-training checkpoints are research/ablation artifacts.

Absolute geodetic elevation is handled by the calibration stage rather than forced
directly from RGB-only prediction.

Map imagery:

- The Geospatial Workspace defaults to Earth Search Sentinel-2 L2A imagery.
- `Esri World Imagery (high-resolution)` is also available as a selectable
  public RGB source and uses the same exact-AOI export path as the map. Its
  availability and detail vary by location; attribution is retained in job
  metadata.
- Optional high-resolution STAC imagery is enabled with
  `ASTERRA_IMAGERY_HIGHRES_STAC_URL` and `ASTERRA_IMAGERY_HIGHRES_COLLECTION`.
- Set `ASTERRA_IMAGERY_ESRI_URL=` to disable the public Esri provider or to
  point it at a permitted ArcGIS export endpoint.
- Map jobs enrich the exported GLB and terrain view with OpenStreetMap building
  footprints. OSM `height`/`building:levels` tags are used when present; high-
  resolution sources can use metric DSM samples, while coarse Sentinel-2
  scenes use clearly labelled approximate context massing.
- The same AOI query adds road ribbons, water/green-area overlays, and
  low-poly tree context to the terrain view and GLB. Tree/woodland heights are
  labelled approximate when the source catalog does not provide measurements.
- Configure the footprint service with `ASTERRA_BUILDINGS_OVERPASS_URL` as a
  comma-separated list of Overpass endpoints, or disable it with
  `ASTERRA_BUILDINGS_ENABLED=false`. Building metadata and
  `buildings.geojson` are stored with each map job.
- `ASTERRA_ENVIRONMENT_ENABLED`, `ASTERRA_ENVIRONMENT_MAX_FEATURES`, and
  `ASTERRA_ENVIRONMENT_MAX_TREES` control the AOI environment layer limits.
- `DW_MAP_MOCK=true` exposes deterministic synthetic imagery for local E2E tests only.
- Restart the backend after changing imagery settings.

Pipeline:
RGB / GeoTIFF
-> preprocessing
-> ASTERRA Stage-5 Urban3D model
-> predicted height / relative geometry
-> calibration using available RPC + DEM + optional GCP
-> metric elevation / DSM
-> 3D terrain mesh / flythrough

Do not overwrite the locked checkpoint during deployment experiments.
