from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
import trimesh


def _remove_isolated_spikes(dsm):
    """Remove isolated DSM spikes while preserving broad hills and ridges."""
    try:
        from scipy.ndimage import median_filter
    except Exception:
        return dsm, 0
    data = dsm.astype(np.float32).filled(np.nan)
    valid = np.isfinite(data)
    if not valid.any():
        return dsm, 0
    fill = float(np.nanmedian(data[valid]))
    safe = np.where(valid, data, fill)
    median = median_filter(safe, size=3, mode="nearest")
    delta = np.abs(safe - median)
    finite_delta = delta[valid]
    threshold = max(2.0, float(np.percentile(finite_delta, 98.5))) if finite_delta.size else 2.0
    spikes = valid & (delta > threshold)
    if not spikes.any():
        return dsm, 0
    safe[spikes] = median[spikes]
    return np.ma.array(safe, mask=np.ma.getmaskarray(dsm)), int(spikes.sum())


def _apply_surface_overrides(dsm, transform, surface_overrides):
    """Flatten measured objects out of the terrain height field.

    A DSM contains roofs and canopy tops, while the terrain mesh should be a
    bare-ground surface when those objects are exported as isolated geometry.
    Applying the overrides after downsampling keeps the terrain and object
    layers on exactly the same browser grid and prevents the overlapping-hill
    artefact seen in the original viewer.
    """
    if not surface_overrides:
        return dsm, 0
    data = dsm.astype(np.float32).filled(np.nan)
    mask = np.ma.getmaskarray(dsm).copy()
    shapes = []
    for item in surface_overrides:
        polygon = item.get("polygon_projected") or []
        if len(polygon) < 3:
            continue
        try:
            elevation = float(item.get("ground_elevation"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(elevation):
            continue
        ring = [list(map(float, point)) for point in polygon]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        shapes.append(({"type": "Polygon", "coordinates": [ring]}, elevation))
    if not shapes:
        return dsm, 0
    try:
        replacement = rasterize(
            shapes,
            out_shape=data.shape,
            transform=transform,
            fill=np.nan,
            dtype="float32",
            all_touched=True,
        )
    except (TypeError, ValueError, rasterio.errors.RasterioIOError):
        return dsm, 0
    selected = np.isfinite(replacement)
    data[selected] = replacement[selected]
    mask[selected] = False
    return np.ma.array(data, mask=mask), int(selected.sum())


def load_downsampled_dsm(dsm_path, target_size=512, surface_overrides=None, terrain_filter=False):
    """
    Load ASTERRA metric DSM and downsample it for browser-friendly
    3D visualization.

    The original GeoTIFF is NOT modified.
    """

    dsm_path = Path(dsm_path)

    with rasterio.open(dsm_path) as src:
        # Never upscale a source DSM. Upsampling a 65x43 overview to 512x512
        # only creates interpolated pixels and makes the model look smoother,
        # not more detailed.
        scale = min(
            1.0,
            target_size / src.width,
            target_size / src.height
        )

        out_width = max(2, int(round(src.width * scale)))
        out_height = max(2, int(round(src.height * scale)))

        dsm = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear,
            masked=True
        )

        transform = src.transform * src.transform.scale(
            src.width / out_width,
            src.height / out_height
        )

        crs = src.crs
        bounds = src.bounds
        nodata = src.nodata

    filtered_pixels = 0
    if terrain_filter:
        dsm, filtered_pixels = _remove_isolated_spikes(dsm)
    dsm, override_pixels = _apply_surface_overrides(dsm, transform, surface_overrides)

    return dsm, transform, crs, bounds, nodata, override_pixels, filtered_pixels


def build_mesh_from_dsm(dsm, transform, z_origin=None):
    """
    Convert a metric DSM raster into a triangular terrain mesh.

    Coordinates:
        X = local Easting
        Y = local Northing
        Z = metric elevation

    The raster's top-left corner is used as the local origin.
    """

    if not np.ma.isMaskedArray(dsm):
        dsm = np.ma.masked_invalid(dsm)

    z = dsm.astype(np.float32).filled(np.nan)

    height, width = z.shape

    # Raster pixel-center coordinates.
    rows, cols = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing="ij"
    )

    # Convert raster indices to projected coordinates.
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e

    # Convert to local coordinates to avoid huge UTM values in WebGL.
    x0 = xs[0, 0]
    y0 = ys[0, 0]

    x_local = xs - x0
    y_local = ys - y0

    valid = np.isfinite(z)
    if not valid.any():
        raise ValueError("DSM contains no finite elevation samples.")
    vertical_origin = float(z_origin) if z_origin is not None else float(np.nanmin(z[valid]))
    z_local = z - vertical_origin

    # Every raster sample becomes a potential vertex.
    vertices = np.column_stack(
        (
            x_local.ravel(),
            y_local.ravel(),
            z_local.ravel(),
        )
    ).astype(np.float32)

    vertex_valid = valid.ravel()

    # Four corners of each raster cell.
    idx = np.arange(height * width).reshape(height, width)

    a = idx[:-1, :-1]
    b = idx[:-1, 1:]
    c = idx[1:, :-1]
    d = idx[1:, 1:]

    cell_valid = (
        valid[:-1, :-1]
        & valid[:-1, 1:]
        & valid[1:, :-1]
        & valid[1:, 1:]
    )

    # Two triangles per valid raster cell.
    faces_1 = np.stack(
        [a[cell_valid], c[cell_valid], b[cell_valid]],
        axis=1
    )

    faces_2 = np.stack(
        [b[cell_valid], c[cell_valid], d[cell_valid]],
        axis=1
    )

    faces = np.vstack((faces_1, faces_2)).astype(np.int64)
    if faces.size == 0:
        raise ValueError("DSM contains no contiguous valid cells for 3D reconstruction.")

    # Remove unused vertices and re-index faces.
    used = np.unique(faces)

    remap = np.full(
        len(vertices),
        -1,
        dtype=np.int64
    )

    remap[used] = np.arange(len(used))

    vertices = vertices[used]
    faces = remap[faces]

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False
    )
    # Trimesh 4.x removed ``remove_degenerate_faces``. Filter by triangle
    # area directly so the backend works across the supported trimesh API
    # versions without depending on a deprecated method.
    if len(mesh.faces):
        triangle_vertices = mesh.vertices[mesh.faces]
        cross = np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        )
        non_degenerate = np.linalg.norm(cross, axis=1) > 1e-10
        if not np.all(non_degenerate):
            mesh.update_faces(non_degenerate)
    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()

    metadata = {
        "origin_x": float(x0),
        "origin_y": float(y0),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "width": int(width),
        "height": int(height),
        "vertical_origin_m": vertical_origin,
    }

    return mesh, metadata


def add_terrain_skirt(mesh, depth=None):
    """Give the terrain surface a small, textured vertical edge."""
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return mesh, False
    edge_counts = {}
    for face in mesh.faces:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(start), int(end))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = [edge for edge, count in edge_counts.items() if count == 1]
    if not boundary:
        return mesh, False
    z_min = float(np.min(mesh.vertices[:, 2]))
    z_max = float(np.max(mesh.vertices[:, 2]))
    skirt_depth = float(depth if depth is not None else max(1.0, (z_max - z_min) * 0.08))
    boundary_vertices = sorted({index for edge in boundary for index in edge})
    bottom_index = {index: len(mesh.vertices) + offset for offset, index in enumerate(boundary_vertices)}
    bottom_vertices = mesh.vertices[boundary_vertices].copy()
    bottom_vertices[:, 2] = z_min - skirt_depth
    side_faces = []
    for a, b in boundary:
        side_faces.append((a, b, bottom_index[a]))
        side_faces.append((b, bottom_index[b], bottom_index[a]))
    mesh = trimesh.Trimesh(
        vertices=np.vstack((mesh.vertices, bottom_vertices)).astype(np.float32),
        faces=np.vstack((mesh.faces, np.asarray(side_faces, dtype=np.int64))),
        process=False,
    )
    return mesh, True


def create_dsm_mesh(dsm_path, target_size=512, z_units="meters", surface_overrides=None):
    dsm, transform, crs, bounds, nodata, override_pixels, filtered_pixels = load_downsampled_dsm(
        dsm_path,
        target_size=target_size,
        surface_overrides=surface_overrides,
        terrain_filter=True,
    )

    mesh, metadata = build_mesh_from_dsm(dsm, transform)
    mesh, has_skirt = add_terrain_skirt(mesh)

    metadata.update(
        {
            "crs": str(crs),
            "bounds": [
                float(bounds.left),
                float(bounds.bottom),
                float(bounds.right),
                float(bounds.top),
            ],
            "resolution_x": float(np.hypot(transform.a, transform.d)),
            "resolution_y": float(np.hypot(transform.b, transform.e)),
            "nodata": None if nodata is None else float(nodata),
            "z_units": z_units,
            "has_skirt": has_skirt,
            "surface_overrides": int(override_pixels),
            "surface_sanitized": bool(override_pixels),
            "isolated_spikes_filtered": int(filtered_pixels),
        }
    )

    return mesh, metadata
