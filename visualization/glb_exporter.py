from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image

import trimesh
from trimesh.visual.texture import TextureVisuals, SimpleMaterial


def load_rgb_texture_image(rgb_path, texture_width=512, texture_height=512):
    """Read and normalize the source RGB raster once for all scene layers."""

    rgb_path = Path(rgb_path)

    with rasterio.open(rgb_path) as src:
        if src.count < 3:
            raise ValueError("RGB texture source must contain at least three bands.")
        rgb = src.read(
            [1, 2, 3],
            out_shape=(3, texture_height, texture_width),
            resampling=Resampling.bilinear,
        )

    # Scientific GeoTIFFs are frequently uint16 reflectance. Direct clipping
    # makes the texture almost uniformly white and hides the scene structure.
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
    for band in range(3):
        values = rgb[:, :, band]
        valid = np.isfinite(values)
        if not valid.any():
            rgb[:, :, band] = 0
            continue
        low_value, high_value = float(values[valid].min()), float(values[valid].max())
        if high_value <= 1.0 and low_value >= 0.0:
            low, high = 0.0, 1.0
        elif high_value <= 255.0 and low_value >= 0.0:
            low, high = 0.0, 255.0
        else:
            low, high = np.percentile(values[valid], (2.0, 98.0)).astype(float)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = low_value, high_value
        rgb[:, :, band] = (values - low) * 255.0 / max(high - low, 1e-6)
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def apply_rgb_texture(mesh, image, x_bounds=None, y_bounds=None):
    """Apply source-aligned UVs to any local-coordinate mesh.

    The terrain and every roof use the same local x/y bounds. This keeps roof
    imagery registered with the terrain instead of normalizing each roof to
    its own bounding box.
    """

    if mesh is None or len(mesh.vertices) == 0:
        return mesh

    x = mesh.vertices[:, 0]
    y = mesh.vertices[:, 1]
    if x_bounds is None:
        x_bounds = (float(x.min()), float(x.max()))
    if y_bounds is None:
        y_bounds = (float(y.min()), float(y.max()))

    x_min, x_max = map(float, x_bounds)
    y_min, y_max = map(float, y_bounds)
    u = (x - x_min) / max(x_max - x_min, 1e-8)
    v = 1.0 - ((y - y_min) / max(y_max - y_min, 1e-8))
    uv = np.column_stack((u, v)).astype(np.float32)
    uv = np.nan_to_num(uv, nan=0.0, posinf=1.0, neginf=0.0)

    material = SimpleMaterial(image=image, diffuse=(255, 255, 255, 255))
    mesh.visual = TextureVisuals(uv=uv, material=material)
    return mesh


def add_rgb_texture(mesh, rgb_path, texture_width=512, texture_height=512):
    """
    Attach RGB imagery to the mesh using UV coordinates.
    """

    image = load_rgb_texture_image(rgb_path, texture_width, texture_height)
    return apply_rgb_texture(mesh, image)


def export_glb(mesh, output_path):
    """
    Export textured mesh as binary GLB.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # The reconstruction code works in raster order (X east, Y north, Z
    # elevation).  glTF/Three.js scenes are conventionally Y-up.  Apply one
    # proper rotation at the export boundary so every consumer gets the same
    # axis contract: X east, Y elevation, Z south.
    y_up = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    if isinstance(mesh, trimesh.Scene):
        seen = set()
        for geometry in mesh.geometry.values():
            if id(geometry) in seen:
                continue
            seen.add(id(geometry))
            geometry.apply_transform(y_up)
    else:
        mesh.apply_transform(y_up)

    mesh.export(
        output_path,
        file_type="glb"
    )

    return output_path
