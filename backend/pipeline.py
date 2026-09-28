
"""The one orchestration path used by every upload job."""

import json
from pathlib import Path
import sys

from .storage import JobStore
from .services.metadata_service import materialize_input, write_gcp_csv, write_preview_assets, inspect_raster
from .services.inference_service import inference_service
from .services.calibration_service import run_calibration, run_gcp_only_calibration, write_relative_dsm
from .services.dem_service import resolve_dem_info
from .services.reconstruction_service import reconstruct
from .services.imagery_service import aoi_payload, prepare_map_scene
from .services.building_service import buildings_geojson, environment_geojson, fetch_map_features
from .services.segmentation_service import add_semantic_features, run_segmentation


def _stage(store, job_id, name, state, progress, sub):
    store.update_stage(job_id, name, state, progress, sub)


def run_pipeline(job_id, store: JobStore, settings):
    if str(settings.project_root) not in sys.path:
        sys.path.insert(0, str(settings.project_root))
    paths = store.paths(job_id)
    job = store.get(job_id)
    stage = "ingest"
    try:
        map_scene = None
        if job.get("source_type") == "map":
            _stage(store, job_id, "ingest", "active", 2, "finding imagery scene")
            map_scene = prepare_map_scene(job, paths.source_upload, settings)
            store.update(
                job_id,
                source_name="map-scene.tif",
                map_scene=map_scene,
            )
            job = store.get(job_id)
            _stage(store, job_id, "ingest", "active", 4, "imagery downloaded and clipped to AOI")
        _stage(store, job_id, "ingest", "active", 5, "validating upload and reading geospatial metadata")
        input_meta = materialize_input(
            paths.source_upload,
            paths.source_raster,
            job.get("gcps", []),
            settings.max_raster_pixels,
            job.get("source_name"),
            min_dimension=settings.min_raster_dimension,
        )
        store.update(job_id, input_metadata={k: v for k, v in input_meta.items() if k not in {"transform", "bounds"}})
        _stage(store, job_id, "ingest", "done", 100, "RGB raster staged" + (" · georeference detected" if input_meta["georeferenced"] else " · relative mode"))

        _stage(store, job_id, "ingest", "active", 7, "running semantic land-cover segmentation")
        segmentation_info = run_segmentation(settings, paths.source_raster, paths.semantic_mask, paths.semantic_preview)
        _stage(
            store,
            job_id,
            "ingest",
            "done",
            100,
            "RGB raster + semantic layers staged"
            if segmentation_info.get("mask_path")
            else "RGB raster staged · semantic fallback/disabled",
        )

        stage = "depth"
        _stage(store, job_id, "depth", "active", 0, "loading ASTERRA Stage 5")
        inference = inference_service.run(
            settings, paths.source_raster, paths.prediction_npy, paths.prediction_tif,
            progress_callback=lambda info: _stage(store, job_id, "depth", "active", int(info["fraction"] * 100), f"tiled inference · {info['completed']}/{info['total']} tiles"),
        )
        _stage(store, job_id, "depth", "done", 100, "nDSM generated")

        stage = "calibrate"
        _stage(store, job_id, "calibrate", "active", 0, "selecting DEM provider")
        dem_info = resolve_dem_info(
            paths.source_raster,
            job.get("dem_provider", settings.dem_provider),
            paths.dem_upload if paths.dem_upload.exists() else (job.get("dem_path") or settings.dem_path),
            settings.project_root,
            cache_root=settings.dem_cache_root,
            online=settings.dem_online,
            timeout=settings.dem_download_timeout,
            max_download_bytes=settings.dem_max_download_bytes,
            progress_callback=lambda message: _stage(store, job_id, "calibrate", "active", 20, message),
        )
        dem_path = dem_info.path if dem_info else None
        calibration = None
        metric = False
        calibration_warning = None
        source_gsd_m = None
        if map_scene and map_scene.get("resolution_m") is not None:
            # This is the sensor's native GSD.  The clipped raster may be
            # resampled to the model minimum, so its GeoTIFF resolution is not
            # a measure of recoverable detail.
            source_gsd_m = float(map_scene["resolution_m"])
        coarse_map_surface = bool(source_gsd_m is not None and source_gsd_m > 5.0)
        # Sentinel-2 and similarly coarse scenes cannot resolve buildings. A
        # monocular nDSM prediction at this scale is mostly texture-driven
        # noise; combining it with a DEM creates the lumpy terrain users see.
        # Keep inference in the pipeline for diagnostics, but use the DEM as
        # the metric surface until a genuinely high-resolution provider is
        # selected. Upload jobs and configured high-res map scenes retain the
        # normal learned DSM path.
        ndsm_weight = 0.0 if coarse_map_surface else 1.0
        gcp_path = None
        if job.get("gcps"):
            reference_crs = input_meta.get("crs")
            if write_gcp_csv(job["gcps"], paths.gcp_csv, reference_crs):
                gcp_path = paths.gcp_csv
        if input_meta["georeferenced"] and dem_path:
            try:
                calibration = run_calibration(
                    paths.prediction_npy, dem_path, paths.source_raster, paths.metric_dsm,
                    gcp_path=gcp_path, metadata_path=paths.calibration_metadata,
                    progress_callback=lambda info: _stage(store, job_id, "calibrate", "active", int(info["fraction"] * 100), info["sub"]),
                    ndsm_weight=ndsm_weight,
                )
                metric = bool(calibration.get("metadata", {}).get("is_metric"))
            except Exception as exc:
                calibration_warning = f"Metric calibration unavailable: {exc}"
                _stage(store, job_id, "calibrate", "active", 80, calibration_warning)
        elif input_meta["georeferenced"] and gcp_path:
            try:
                calibration = run_gcp_only_calibration(
                    paths.prediction_tif, paths.source_raster, paths.metric_dsm,
                    gcp_path, paths.calibration_metadata,
                    progress_callback=lambda info: _stage(store, job_id, "calibrate", "active", int(info["fraction"] * 100), info["sub"]),
                )
                metric = bool(calibration.get("metadata", {}).get("is_metric"))
            except Exception as exc:
                calibration_warning = f"GCP calibration unavailable: {exc}"
                _stage(store, job_id, "calibrate", "active", 80, calibration_warning)
        else:
            calibration_warning = "No overlapping DEM or valid GCP calibration was available."
        if not metric:
            calibration = write_relative_dsm(paths.prediction_tif, paths.relative_dsm, paths.calibration_metadata)
        _stage(store, job_id, "calibrate", "done", 100, "metric DSM generated" if metric else f"relative/rDSM · {calibration_warning}")

        stage = "mesh"
        _stage(store, job_id, "mesh", "active", 0, "building textured 3D mesh")
        dsm_path = Path(calibration["output_path"])
        buildings = []
        building_info = {"enabled": False, "provider": "openstreetmap", "count": 0}
        environment = {"roads": [], "water": [], "landcover": [], "trees": [], "semantic_regions": []}
        environment_info = {"enabled": False, "provider": "openstreetmap"}
        semantic_feature_info = {"regions_count": 0, "building_candidates": 0}
        if map_scene and (settings.buildings_enabled or settings.environment_enabled):
            _stage(store, job_id, "mesh", "active", 8, "loading buildings, roads, water, and trees")
            buildings, building_info, environment, environment_info = fetch_map_features(
                job["aoi"],
                paths.source_raster,
                dsm_path,
                endpoint=settings.buildings_overpass_url,
                timeout=settings.buildings_timeout,
                max_buildings=settings.buildings_max if settings.buildings_enabled else 0,
                max_features=settings.environment_max_features if settings.environment_enabled else 0,
                max_trees=settings.environment_max_trees if settings.environment_enabled else 0,
                min_area_m2=settings.buildings_min_area_m2,
                source_gsd_m=source_gsd_m,
                mask_path=paths.semantic_mask,
            )
        # Semantic extraction is also useful for georeferenced uploads. OSM
        # remains preferred where it is available, while the deterministic
        # mask supplies separate evidence layers and bounded candidates.
        if paths.semantic_mask.exists():
            environment, buildings, semantic_feature_info = add_semantic_features(
                environment,
                buildings,
                paths.semantic_mask,
                dsm_path,
            )
            environment_info = {
                **environment_info,
                "semantic_regions_count": semantic_feature_info["regions_count"],
                "semantic_building_candidates": semantic_feature_info["building_candidates"],
                "semantic_approximate_tree_count": semantic_feature_info.get("approximate_tree_count", 0),
            }
        if map_scene or buildings or any(environment.values()):
            paths.buildings_geojson.write_text(
                json.dumps(buildings_geojson(buildings, input_meta.get("crs")), indent=2),
                encoding="utf-8",
            )
            paths.environment_geojson.write_text(
                json.dumps(environment_geojson(environment, input_meta.get("crs")), indent=2),
                encoding="utf-8",
            )
        if segmentation_info.get("fallback") and segmentation_info.get("reason"):
            environment_info = {**environment_info, "semantic_warning": segmentation_info["reason"]}
        reconstruction_gsd_m = source_gsd_m
        if reconstruction_gsd_m is None:
            pixel_size = input_meta.get("pixel_size_m") or input_meta.get("resolution") or [None, None]
            try:
                candidate_gsd = (float(pixel_size[0]) + float(pixel_size[1])) / 2.0
                reconstruction_gsd_m = candidate_gsd if candidate_gsd > 0 else None
            except (TypeError, ValueError, IndexError):
                reconstruction_gsd_m = None
        reconstruction = reconstruct(
            dsm_path, paths.source_raster, paths.model_glb, paths.reconstruction_metadata,
            size=settings.mesh_size,
            progress_callback=lambda info: _stage(store, job_id, "mesh", "active", int(info["fraction"] * 100), info["sub"]),
            z_units="meters" if metric else "relative",
            buildings=buildings,
            environment=environment,
            source_gsd_m=reconstruction_gsd_m,
        )
        write_preview_assets(paths.source_raster, dsm_path, paths.texture, paths.normal)
        _stage(store, job_id, "mesh", "done", 100, "GLB exported")

        pixel_size = input_meta.get("pixel_size_m") or input_meta.get("resolution") or [1.0, 1.0]
        try:
            gsd_m = float((float(pixel_size[0]) + float(pixel_size[1])) / 2.0)
        except (TypeError, ValueError, IndexError):
            gsd_m = None
        if source_gsd_m is not None:
            gsd_m = source_gsd_m
        quality_warnings = []
        terrain_only = False
        if map_scene and map_scene.get("fallback_warning"):
            quality_warnings.append(map_scene["fallback_warning"])
        elif map_scene and map_scene.get("fallback_chain"):
            quality_warnings.append(
                "Imagery fallback chain used: "
                + " → ".join(str(value) for value in map_scene["fallback_chain"])
            )
        if gsd_m is not None and gsd_m > 5:
            terrain_only = True
            quality_warnings.append(f"Source GSD is {gsd_m:.2f} m/pixel; output is terrain-scale and not building-level.")
            if metric and coarse_map_surface:
                quality_warnings.append("Coarse imagery is rendered against the DEM terrain surface; no building detail is inferred from upsampled pixels.")
        if gsd_m is not None and gsd_m > 10:
            quality_warnings.append("Building-level height measurements are disabled for this source resolution.")
        has_environment = any(
            bool(values)
            for key, values in environment.items()
            if key != "semantic_regions"
        )
        has_features = bool(buildings) or has_environment
        approximate_geometry_count = sum(
            1 for building in buildings
            if str(building.get("geometry_quality", "")).lower() == "approximate"
        )
        has_approximate_features = bool(
            building_info.get("approximate_count")
            or approximate_geometry_count
            or environment_info.get("approximate_tree_count")
            or environment_info.get("semantic_approximate_tree_count")
            or (segmentation_info.get("fallback") and (
                semantic_feature_info.get("regions_count") or semantic_feature_info.get("building_candidates")
            ))
            or (map_scene and map_scene.get("fallback_warning"))
        )
        if not has_features:
            scene_quality = "terrain_only"
        elif has_approximate_features or terrain_only:
            scene_quality = "hybrid"
        else:
            scene_quality = "measured"
        calibration_metadata = {
            **calibration.get("metadata", {}),
            "status": "metric" if metric else "relative",
            "provider": dem_info.metadata.get("provider") if dem_info else None,
            "source": dem_info.metadata.get("source") if dem_info else None,
            "units": "meters" if metric else "relative",
            "vertical_reference": calibration.get("metadata", {}).get("vertical_reference") or (dem_info.metadata.get("vertical_reference") if dem_info else None),
            "warning": calibration_warning,
            "surface_mode": calibration.get("metadata", {}).get("surface_mode") or ("terrain-dem" if coarse_map_surface and metric else "dsm"),
            "ndsm_weight": calibration.get("metadata", {}).get("ndsm_weight", ndsm_weight),
        }
        for warning in (
            building_info.get("warning"),
            environment_info.get("warning"),
            environment_info.get("semantic_warning"),
        ):
            if warning and warning not in quality_warnings:
                quality_warnings.append(warning)
        reconstruction_metadata = {
            **reconstruction["metadata"],
            "mode": "dsm-terrain",
            "has_glb": True,
            "glb_url": f"/api/jobs/{job_id}/export/model.glb",
            "building_detection": (
                f"{building_info.get('provider', 'mapped')} footprints + metric/approximate heights"
                if buildings
                else ("unavailable" if building_info.get("warning") else "not_available")
            ),
            "buildings_detected": len(buildings),
            "geometry_source": "metric DSM height field" if metric else "relative nDSM height field",
            "terrain_only": terrain_only,
            "scene_quality": scene_quality,
            "quality_warnings": quality_warnings,
            "calibration_surface_mode": calibration_metadata.get("surface_mode"),
            "building_source": building_info,
            "has_buildings": bool(buildings),
            "has_geojson": paths.buildings_geojson.exists(),
            "segmentation": segmentation_info,
            "has_mask": paths.semantic_preview.exists(),
            "mask_url": f"/api/jobs/{job_id}/export/mask.png" if paths.semantic_preview.exists() else None,
            "environment": environment_info,
            "has_environment": has_environment,
            "has_environment_geojson": paths.environment_geojson.exists(),
            "surface_mode": reconstruction.get("metadata", {}).get("reconstruction", {}).get("surface_mode", "hybrid-dsm-semantic"),
            "roof_measured": reconstruction.get("metadata", {}).get("reconstruction", {}).get("roof_measured", 0),
            "roof_approximate": reconstruction.get("metadata", {}).get("reconstruction", {}).get("roof_approximate", 0),
            "semantic_backend": segmentation_info.get("model") or ("deterministic-rgb-fallback" if segmentation_info.get("fallback") else "unknown"),
            "semantic_layer_counts": reconstruction.get("metadata", {}).get("environment_counts", {}).get("semantic_layers", {}),
        }
        source = {"type": "upload", "name": job.get("source_name")}
        georeference = {
            "crs": input_meta.get("crs"),
            "bounds": input_meta.get("bounds"),
            "resolution": input_meta.get("resolution"),
        } if input_meta.get("georeferenced") else None
        if map_scene:
            source = {
                "type": "map",
                "name": job.get("source_name"),
                "aoi": aoi_payload(job["aoi"]),
                "imagery": map_scene,
            }
            georeference = {
                "crs": input_meta.get("crs"),
                "bounds": {
                    "north": job["aoi"]["north"],
                    "south": job["aoi"]["south"],
                    "east": job["aoi"]["east"],
                    "west": job["aoi"]["west"],
                },
                "origin": {"lat": job["aoi"]["north"], "lon": job["aoi"]["west"]},
                "resolution": input_meta.get("resolution"),
                "raster_bounds": input_meta.get("bounds"),
            }
        aggregate = {
            "job_id": job_id, "source": source,
            "model": inference["model"], "device": inference["device"], "metric": metric,
            "is_metric": metric, "metric_valid": metric, "crs": calibration.get("crs") or input_meta.get("crs"),
            "width": calibration["width"], "height": calibration["height"], "stats": calibration["stats"],
            "metadata": {
                **calibration_metadata,
                "pixel_size_m": input_meta.get("pixel_size_m"),
                "input_reprojected_to_metric_crs": input_meta.get("reprojected_to_metric_crs", False),
                "input_dtype_normalized": input_meta.get("input_dtype_normalized", False),
            },
            "calibration": calibration_metadata,
            "quality": {"gsd_m": gsd_m, "terrain_only": terrain_only, "scene_quality": scene_quality, "warnings": quality_warnings},
            "reconstruction": reconstruction_metadata,
            "buildings": buildings,
            "building_source": building_info,
            "environment": environment,
            "environment_source": environment_info,
            "segmentation": segmentation_info,
            "georeference": georeference,
            "artifacts": {"dsm": str(dsm_path), "glb": str(paths.model_glb), "metadata": str(paths.root / "metadata.json"), "environment": str(paths.environment_geojson), "mask": str(paths.semantic_preview)},
        }
        aggregate_path = paths.root / "metadata.json"
        aggregate_path.write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
        result = {
            "metric": metric, "is_metric": metric, "metric_valid": metric, "crs": aggregate["crs"],
            "width": aggregate["width"], "height": aggregate["height"], "stats": aggregate["stats"],
            "model": aggregate["model"], "metadata": aggregate["metadata"], "calibration": aggregate["calibration"], "quality": aggregate["quality"], "reconstruction": aggregate["reconstruction"], "buildings": aggregate["buildings"], "building_source": aggregate["building_source"], "environment": aggregate["environment"], "environment_source": aggregate["environment_source"],
            "source": aggregate["source"], "georeference": aggregate["georeference"],
            "dsm_url": f"/api/jobs/{job_id}/export/dsm.tif", "glb_url": f"/api/jobs/{job_id}/export/model.glb",
            "metadata_url": f"/api/jobs/{job_id}/export/metadata.json", "texture_url": f"/files/{job_id}/texture.jpg",
            "normal_url": f"/files/{job_id}/normal.png", "mask_url": aggregate["reconstruction"].get("mask_url"), "file_token": job["file_token"],
        }
        store.complete(job_id, result)
    except Exception:
        raise
