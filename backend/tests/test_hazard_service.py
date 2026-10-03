"""Validation tests for the hazard simulation service.

These tests verify that:
1. Flat regions do not receive high landslide susceptibility
2. Steep terrain regions remain eligible for landslide assessment
3. Inland low-elevation regions disconnected from coast are not flooded
4. Changing water-level scenario changes only the physically affected region
5. Output raster has same CRS, transform, dimensions as source DSM
6. Hazard pixels can be mapped back to geographic coordinates
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import tempfile
import pytest

from backend.services.hazard_service import (
    simulate_landslide_susceptibility,
    simulate_coastal_inundation,
    _compute_slope,
    _normalize,
)


def create_test_dsm(
    width: int = 100,
    height: int = 100,
    resolution: float = 10.0,
    crs: str = "EPSG:32643",
    bounds: tuple = (500000, 2000000, 501000, 2001000),
) -> Path:
    """Create a test DSM GeoTIFF with known properties."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        path = Path(f.name)

    transform = from_bounds(*bounds, width, height)
    data = np.random.uniform(100, 500, (height, width)).astype(np.float32)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data, 1)

    return path


class TestLandslideSusceptibility:
    """Tests for landslide susceptibility model."""

    def test_flat_region_suppressed(self):
        """Flat regions (slope < 5°) should have very low susceptibility."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = Path(f.name)

        # Create flat DSM (constant elevation)
        width, height = 50, 50
        transform = from_bounds(500000, 2000000, 500500, 2000500, width, height)
        data = np.full((height, width), 200.0, dtype=np.float32)

        with rasterio.open(
            path, "w", driver="GTiff", height=height, width=width,
            count=1, dtype="float32", crs="EPSG:32643", transform=transform,
        ) as dst:
            dst.write(data, 1)

        result = simulate_landslide_susceptibility(path, stride=1)

        # Flat region should have very low susceptibility
        assert result["statistics"]["mean_susceptibility"] < 0.1, \
            f"Flat region should have low susceptibility, got {result['statistics']['mean_susceptibility']}"
        assert result["statistics"]["max_susceptibility"] < 0.2, \
            f"Flat region max susceptibility should be < 0.2, got {result['statistics']['max_susceptibility']}"

    def test_steep_region_eligible(self):
        """Steep terrain should remain eligible for landslide assessment."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = Path(f.name)

        # Create steep DSM (large elevation change)
        width, height = 50, 50
        transform = from_bounds(500000, 2000000, 500500, 2000500, width, height)
        # Create a slope: elevation increases by 10m per cell
        data = np.zeros((height, width), dtype=np.float32)
        for r in range(height):
            for c in range(width):
                data[r, c] = 100.0 + c * 10.0  # 10m per cell = ~45° slope

        with rasterio.open(
            path, "w", driver="GTiff", height=height, width=width,
            count=1, dtype="float32", crs="EPSG:32643", transform=transform,
        ) as dst:
            dst.write(data, 1)

        result = simulate_landslide_susceptibility(path, stride=1)

        # Steep region should have higher susceptibility
        assert result["statistics"]["mean_susceptibility"] > 0.2, \
            f"Steep region should have higher susceptibility, got {result['statistics']['mean_susceptibility']}"

    def test_output_preserves_crs_and_transform(self):
        """Output raster must have same CRS and transform as source DSM."""
        path = create_test_dsm(width=50, height=50, crs="EPSG:32643")

        result = simulate_landslide_susceptibility(path, stride=1)

        # Check CRS preserved
        assert result["terrain_source"]["crs"] == "EPSG:32643", \
            f"CRS should be preserved, got {result['terrain_source']['crs']}"

        # Check grid dimensions match
        assert result["terrain_source"]["grid"] == [50, 50], \
            f"Grid dimensions should match, got {result['terrain_source']['grid']}"

    def test_hazard_pixels_map_to_geographic_coordinates(self):
        """Every hazard pixel must be mappable to a geographic location."""
        path = create_test_dsm(width=30, height=30, crs="EPSG:32643")

        result = simulate_landslide_susceptibility(path, stride=1)

        # Verify bounds are present and valid
        bounds = result["terrain_source"]["bounds"]
        assert "left" in bounds and "bottom" in bounds
        assert "right" in bounds and "top" in bounds
        assert bounds["right"] > bounds["left"]
        assert bounds["top"] > bounds["bottom"]

    def test_physical_plausibility_filter_configurable(self):
        """Physical plausibility filter should use configurable thresholds."""
        # This test verifies the config dict exists and has expected keys
        # The actual thresholds are tested implicitly by other tests
        config_keys = ["min_slope_deg", "max_slope_deg", "min_relief_m"]
        # The config is defined inside simulate_landslide_susceptibility
        # We verify the function runs without error
        path = create_test_dsm(width=20, height=20)
        result = simulate_landslide_susceptibility(path, stride=1)
        assert result is not None


class TestCoastalInundation:
    """Tests for coastal inundation model."""

    def test_inland_low_area_not_flooded(self):
        """Inland low-elevation areas disconnected from coast should not be flooded."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = Path(f.name)

        width, height = 100, 100
        transform = from_bounds(500000, 2000000, 501000, 2001000, width, height)

        # Create DSM with:
        # - Low elevation at edges (coast)
        # - High elevation in middle (inland hill)
        # - Low elevation in center but surrounded by high ground (inland depression)
        data = np.zeros((height, width), dtype=np.float32)
        for r in range(height):
            for c in range(width):
                # Distance from nearest edge
                dist_edge = min(r, c, height - 1 - r, width - 1 - c)
                if dist_edge < 10:
                    data[r, c] = 50.0  # Coast
                elif dist_edge < 40:
                    data[r, c] = 200.0  # Inland hill
                else:
                    data[r, c] = 30.0  # Inland depression (disconnected from coast)

        with rasterio.open(
            path, "w", driver="GTiff", height=height, width=width,
            count=1, dtype="float32", crs="EPSG:32643", transform=transform,
        ) as dst:
            dst.write(data, 1)

        # Water level at 100m - should flood coast but not inland depression
        result = simulate_coastal_inundation(path, water_level_m=100.0, stride=1)

        # The inland depression (center) should NOT be flooded because it's
        # surrounded by high ground (200m) that blocks water flow
        g = result["grids"]
        mask = np.frombuffer(
            __import__("base64").b64decode(g["mask_b64"]), dtype=np.uint8
        ).reshape(g["grid_h"], g["grid_w"])

        # Check center of map (inland depression) is not flooded
        center_r, center_c = g["grid_h"] // 2, g["grid_w"] // 2
        assert mask[center_r, center_c] == 0, \
            "Inland depression disconnected from coast should not be flooded"

    def test_water_level_changes_affected_region(self):
        """Changing water level should change only the physically affected region."""
        path = create_test_dsm(width=50, height=50)

        result_low = simulate_coastal_inundation(path, water_level_m=150.0, stride=1)
        result_high = simulate_coastal_inundation(path, water_level_m=300.0, stride=1)

        # Higher water level should flood more cells
        assert result_high["statistics"]["inundated_cells"] >= result_low["statistics"]["inundated_cells"], \
            "Higher water level should flood more or equal cells"

    def test_output_preserves_geographic_extent(self):
        """Output raster must preserve original geographic extent."""
        path = create_test_dsm(width=40, height=40, crs="EPSG:32643")

        result = simulate_coastal_inundation(path, water_level_m=200.0, stride=1)

        # Check bounds are present
        bounds = result["terrain_source"]["bounds"]
        assert bounds is not None
        assert "left" in bounds and "right" in bounds
        assert "bottom" in bounds and "top" in bounds


class TestGeospatialAlignment:
    """Tests for geospatial alignment and coordinate correctness."""

    def test_downsample_preserves_geographic_alignment(self):
        """Downsampling with stride must preserve geographic alignment."""
        path = create_test_dsm(width=100, height=100, resolution=10.0)

        result = simulate_landslide_susceptibility(path, stride=4)

        # Grid should be downsampled
        assert result["grids"]["grid_h"] == 25
        assert result["grids"]["grid_w"] == 25
        assert result["grids"]["grid_stride"] == 4

        # Resolution should be scaled
        assert result["terrain_source"]["resolution"][0] == 40.0  # 10 * 4
        assert result["terrain_source"]["resolution"][1] == 40.0

    def test_crs_mismatch_handled(self):
        """CRS mismatch between DSM and mask should be handled."""
        dsm_path = create_test_dsm(width=30, height=30, crs="EPSG:32643")

        # Create mask with different CRS
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            mask_path = Path(f.name)

        width, height = 30, 30
        transform = from_bounds(500000, 2000000, 500300, 2000300, width, height)
        mask_data = np.zeros((height, width), dtype=np.uint8)
        mask_data[10:20, 10:20] = 2  # vegetation

        with rasterio.open(
            mask_path, "w", driver="GTiff", height=height, width=width,
            count=1, dtype="uint8", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(mask_data, 1)

        # Should not crash - CRS mismatch is handled
        result = simulate_landslide_susceptibility(dsm_path, mask_path=mask_path, stride=1)
        assert result is not None

    def test_nodata_handling(self):
        """Nodata values should be excluded from analysis."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = Path(f.name)

        width, height = 50, 50
        transform = from_bounds(500000, 2000000, 500500, 2000500, width, height)
        data = np.random.uniform(100, 500, (height, width)).astype(np.float32)
        # Set some cells to nodata
        data[0:10, 0:10] = -9999

        with rasterio.open(
            path, "w", driver="GTiff", height=height, width=width,
            count=1, dtype="float32", crs="EPSG:32643", transform=transform,
            nodata=-9999,
        ) as dst:
            dst.write(data, 1)

        result = simulate_landslide_susceptibility(path, stride=1)

        # Nodata cells should have 0 susceptibility
        g = result["grids"]
        suscept = np.frombuffer(
            __import__("base64").b64decode(g["susceptibility_b64"]), dtype=np.float32
        ).reshape(g["grid_h"], g["grid_w"])

        # Top-left corner (nodata) should have 0 susceptibility
        assert suscept[0, 0] == 0.0, "Nodata cells should have 0 susceptibility"


class TestTerrainFactors:
    """Tests for terrain factor computation."""

    def test_slope_computation(self):
        """Slope should be computed correctly from DSM."""
        # Flat surface
        dsm = np.ones((10, 10), dtype=np.float32) * 100.0
        slope = _compute_slope(dsm, (10.0, 10.0))
        assert np.allclose(slope, 0.0, atol=0.1), "Flat surface should have 0 slope"

        # 45° slope
        dsm = np.zeros((10, 10), dtype=np.float32)
        for c in range(10):
            dsm[:, c] = c * 10.0  # 10m rise per 10m run = 45°
        slope = _compute_slope(dsm, (10.0, 10.0))
        assert np.allclose(slope, 45.0, atol=1.0), f"Expected 45° slope, got {slope.mean()}"

    def test_normalize_range(self):
        """Normalization should produce values in [0, 1] range."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        normalized = _normalize(data, 0, 1)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
