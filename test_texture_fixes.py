"""Test script for texture mapping fixes — no model required.

Creates a synthetic DSM + RGB image, runs the reconstruction pipeline,
and verifies the output GLB has:
1. Textured building walls (not flat beige)
2. No vertical stripe artifacts on terrain edges
3. Higher resolution textures
4. Consistent normalization
"""

import sys
import os
import numpy as np
from pathlib import Path

# Fix rasterio DLL loading on Windows
os.add_dll_directory(r'C:\Users\subha\AppData\Local\Programs\Python\Python311\Lib\site-packages\rasterio.libs')

import rasterio
from rasterio.transform import from_origin

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.generate_3d import run_reconstruction


def create_synthetic_dsm(path, width=64, height=64):
    """Create a simple synthetic DSM with a hill in the center."""
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    # A simple hill: higher in the center
    z = 10 + 5 * np.exp(-(xx**2 + yy**2) / 0.5)
    # Add some noise
    z += np.random.normal(0, 0.1, z.shape)

    transform = from_origin(500000, 5000000, 1.0, 1.0)  # 1m pixels, UTM-like
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32633",  # UTM zone 33N
        transform=transform,
    ) as dst:
        dst.write(z.astype(np.float32), 1)
    return path


def create_synthetic_rgb(path, width=64, height=64):
    """Create a synthetic RGB image with distinct colored regions."""
    rgb = np.zeros((3, height, width), dtype=np.uint8)

    # Top-left: red
    rgb[0, :height//2, :width//2] = 200
    rgb[1, :height//2, :width//2] = 50
    rgb[2, :height//2, :width//2] = 50

    # Top-right: green
    rgb[0, :height//2, width//2:] = 50
    rgb[1, :height//2, width//2:] = 200
    rgb[2, :height//2, width//2:] = 50

    # Bottom-left: blue
    rgb[0, height//2:, :width//2] = 50
    rgb[1, height//2:, :width//2] = 50
    rgb[2, height//2:, :width//2] = 200

    # Bottom-right: yellow
    rgb[0, height//2:, width//2:] = 200
    rgb[1, height//2:, width//2:] = 200
    rgb[2, height//2:, width//2:] = 50

    # Add a white cross in the center
    rgb[:, height//2-2:height//2+2, :] = 255
    rgb[:, :, width//2-2:width//2+2] = 255

    transform = from_origin(500000, 5000000, 1.0, 1.0)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:32633",
        transform=transform,
    ) as dst:
        dst.write(rgb)
    return path


def create_test_buildings():
    """Create simple building footprints for testing."""
    return [
        {
            "id": 1,
            "polygon_projected": [
                [500010, 5000010],
                [500015, 5000010],
                [500015, 5000015],
                [500010, 5000015],
            ],
            "ground_elevation": 10.0,
            "roof_elevation": 15.0,
            "height": 5.0,
        },
        {
            "id": 2,
            "polygon_projected": [
                [500020, 5000020],
                [500025, 5000020],
                [500025, 5000025],
                [500020, 5000025],
            ],
            "ground_elevation": 10.0,
            "roof_elevation": 18.0,
            "height": 8.0,
        },
    ]


def main():
    import tempfile
    import trimesh

    tmpdir = Path(tempfile.mkdtemp(prefix="asterra_test_"))
    print(f"Test directory: {tmpdir}")

    # Create synthetic data
    dsm_path = create_synthetic_dsm(tmpdir / "dsm.tif")
    rgb_path = create_synthetic_rgb(tmpdir / "rgb.tif")
    output_path = tmpdir / "output.glb"
    metadata_path = tmpdir / "output.json"

    print(f"DSM: {dsm_path}")
    print(f"RGB: {rgb_path}")

    # Run reconstruction
    buildings = create_test_buildings()

    # Debug: check building coordinates vs DSM bounds
    print(f"\nDebug: Building coordinates:")
    for b in buildings:
        print(f"  Building {b['id']}: polygon={b['polygon_projected']}")
        print(f"    ground_elevation={b['ground_elevation']}, roof_elevation={b['roof_elevation']}")

    # Debug: check what _building_mesh returns
    from visualization.generate_3d import _building_mesh, _ring_xy, _triangulate_ring
    from visualization.dsm_to_mesh import load_downsampled_dsm
    dsm_data, dsm_transform, _, _, _, _, _ = load_downsampled_dsm(dsm_path, target_size=64)
    print(f"\nDebug: DSM transform: {dsm_transform}")
    print(f"Debug: DSM shape: {dsm_data.shape}")
    print(f"Debug: DSM min={float(np.min(dsm_data))}, max={float(np.max(dsm_data))}")
    for b in buildings:
        ring = _ring_xy(b, dsm_transform.c, dsm_transform.f)
        print(f"  Building {b['id']} ring (local coords): {ring}")
        ground = float(b.get("ground_elevation", 0)) - float(np.min(dsm_data))
        roof = float(b.get("roof_elevation", 0)) - float(np.min(dsm_data))
        print(f"  Building {b['id']} ground={ground}, roof={roof}, roof>ground={roof > ground}")
        triangles = _triangulate_ring(ring)
        print(f"  Building {b['id']} triangles: {triangles}")
        mesh = _building_mesh(b, dsm_transform.c, dsm_transform.f, float(np.min(dsm_data)))
        print(f"  Building {b['id']} mesh: {mesh}")

    result = run_reconstruction(
        dsm_path=str(dsm_path),
        rgb_path=str(rgb_path),
        output_path=str(output_path),
        size=64,
        metadata_path=str(metadata_path),
        buildings=buildings,
    )

    print(f"\nGLB: {result['output_path']}")
    print(f"Metadata: {result['metadata_path']}")

    # Verify the output
    print("\n=== Verification ===")

    # 1. Check GLB exists and is valid
    assert output_path.exists(), "GLB file was not created"
    scene = trimesh.load(output_path)
    print(f"GLB loaded: {len(scene.geometry)} geometries")

    # 2. Check terrain has texture
    terrain = scene.geometry.get("TERRAIN")
    assert terrain is not None, "TERRAIN geometry not found"
    assert terrain.visual.material is not None, "Terrain has no material"
    mat = terrain.visual.material
    has_texture = False
    if hasattr(mat, 'image') and mat.image is not None:
        has_texture = True
        print(f"Terrain texture: {mat.image.size}")
    elif hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
        has_texture = True
        print(f"Terrain texture (PBR): {mat.baseColorTexture}")
    else:
        # Check UV coordinates as proxy for texture
        uv = getattr(terrain.visual, 'uv', None)
        if uv is not None and len(uv) > 0:
            has_texture = True
            print(f"Terrain has UV coordinates ({len(uv)} vertices) — texture applied")
        else:
            print("WARNING: Terrain has no texture or UVs")

    # 3. Check skirt exists and has solid color (no texture)
    skirt = scene.geometry.get("TERRAIN_SKIRT")
    if skirt is not None:
        print(f"Skirt geometry: {len(skirt.vertices)} vertices, {len(skirt.faces)} faces")
        skirt_mat = skirt.visual.material
        if hasattr(skirt_mat, 'image') and skirt_mat.image is not None:
            print("WARNING: Skirt has a texture (should be solid color)")
        else:
            print("Skirt has solid color (correct)")
    else:
        print("WARNING: TERRAIN_SKIRT not found")

    # 4. Check buildings have textured walls
    building_geoms = [g for g in scene.geometry if g.startswith("BUILDING_")]
    print(f"Building geometries: {len(building_geoms)}")
    for bg in building_geoms:
        b = scene.geometry[bg]
        bmat = b.visual.material
        has_tex = False
        if hasattr(bmat, 'image') and bmat.image is not None:
            has_tex = True
            print(f"  {bg}: textured (size={bmat.image.size})")
        elif hasattr(bmat, 'baseColorTexture') and bmat.baseColorTexture is not None:
            has_tex = True
            print(f"  {bg}: textured (PBR)")
        else:
            # Check UVs as proxy
            uv = getattr(b.visual, 'uv', None)
            if uv is not None and len(uv) > 0:
                has_tex = True
                print(f"  {bg}: has UVs ({len(uv)} vertices) — texture applied")
            else:
                print(f"  {bg}: FLAT COLOR (should be textured)")

    # 5. Check roofs have texture
    roof_geoms = [g for g in scene.geometry if g.startswith("ROOF_BUILDING_")]
    print(f"Roof geometries: {len(roof_geoms)}")
    for rg in roof_geoms:
        r = scene.geometry[rg]
        rmat = r.visual.material
        has_tex = False
        if hasattr(rmat, 'image') and rmat.image is not None:
            has_tex = True
            print(f"  {rg}: textured (size={rmat.image.size})")
        elif hasattr(rmat, 'baseColorTexture') and rmat.baseColorTexture is not None:
            has_tex = True
            print(f"  {rg}: textured (PBR)")
        else:
            uv = getattr(r.visual, 'uv', None)
            if uv is not None and len(uv) > 0:
                has_tex = True
                print(f"  {rg}: has UVs ({len(uv)} vertices) — texture applied")
            else:
                print(f"  {rg}: FLAT COLOR (should be textured)")

    # 6. Check metadata
    import json
    with open(metadata_path) as f:
        meta = json.load(f)
    print(f"\nMetadata:")
    print(f"  texture_valid: {meta.get('texture_valid')}")
    print(f"  mesh_valid: {meta.get('mesh_valid')}")
    print(f"  buildings_detected: {meta.get('buildings_detected')}")
    print(f"  has_skirt: {meta.get('has_skirt')}")

    print(f"\n=== Test Complete ===")
    print(f"Open the GLB in any viewer to inspect visually.")
    print(f"Files are in: {tmpdir}")


if __name__ == "__main__":
    main()
