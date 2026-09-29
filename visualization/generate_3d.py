import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import trimesh
from rasterio.features import geometry_mask
from trimesh import repair as trimesh_repair
from trimesh.visual.material import SimpleMaterial

try:
    from .dsm_to_mesh import create_dsm_mesh, load_downsampled_dsm
    from .glb_exporter import apply_rgb_texture, export_glb, load_rgb_texture_image
except ImportError:
    from dsm_to_mesh import create_dsm_mesh, load_downsampled_dsm
    from glb_exporter import apply_rgb_texture, export_glb, load_rgb_texture_image


def _ring_xy(building, origin_x, origin_y):
    polygon = building.get("polygon_projected") or []
    if len(polygon) < 3:
        return []
    ring = [(float(point[0]) - origin_x, float(point[1]) - origin_y) for point in polygon]
    # Raster-derived polygons frequently contain repeated and almost-collinear
    # points.  Removing them before ear clipping prevents self-intersections
    # and skinny triangles that read as malformed building slabs in WebGL.
    clean = []
    for point in ring:
        if not clean or np.hypot(point[0] - clean[-1][0], point[1] - clean[-1][1]) > 1e-5:
            clean.append(point)
    if len(clean) > 1 and np.hypot(clean[0][0] - clean[-1][0], clean[0][1] - clean[-1][1]) < 1e-5:
        clean.pop()
    changed = True
    while changed and len(clean) > 3:
        changed = False
        reduced = []
        for index, point in enumerate(clean):
            prev = clean[index - 1]
            nxt = clean[(index + 1) % len(clean)]
            cross = (point[0] - prev[0]) * (nxt[1] - point[1]) - (point[1] - prev[1]) * (nxt[0] - point[0])
            if abs(cross) < 1e-6:
                changed = True
            else:
                reduced.append(point)
        clean = reduced if len(reduced) >= 3 else clean
    ring = clean
    signed_area = sum(
        ring[index][0] * ring[(index + 1) % len(ring)][1]
        - ring[(index + 1) % len(ring)][0] * ring[index][1]
        for index in range(len(ring))
    ) * 0.5
    if signed_area < 0:
        ring.reverse()
    return ring


def _triangulate_ring(ring):
    """Dependency-free ear clipping for cleaned concave footprints."""
    if len(ring) < 3:
        return []
    points = np.asarray(ring, dtype=np.float64)
    area = 0.5 * np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - np.roll(points[:, 0], -1) * points[:, 1]
    )
    indices = list(range(len(points))) if area >= 0 else list(range(len(points) - 1, -1, -1))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])

    def inside(point, a, b, c):
        c1 = cross(a, b, point)
        c2 = cross(b, c, point)
        c3 = cross(c, a, point)
        return c1 >= -1e-8 and c2 >= -1e-8 and c3 >= -1e-8

    triangles = []
    guard = 0
    while len(indices) > 3 and guard < len(points) * len(points):
        guard += 1
        clipped = False
        for pos in range(len(indices)):
            prev_i = indices[pos - 1]
            curr_i = indices[pos]
            next_i = indices[(pos + 1) % len(indices)]
            a, b, c = points[prev_i], points[curr_i], points[next_i]
            if cross(a, b, c) <= 1e-8:
                continue
            if any(
                other not in (prev_i, curr_i, next_i)
                and inside(points[other], a, b, c)
                for other in indices
            ):
                continue
            triangles.append((prev_i, curr_i, next_i))
            indices.pop(pos)
            clipped = True
            break
        if not clipped:
            return []
    if len(indices) == 3:
        triangles.append(tuple(indices))
    return triangles


def _roof_surface_mesh(
    building,
    raw_dsm,
    transform,
    origin_x,
    origin_y,
    vertical_origin,
    source_gsd_m=None,
    min_valid_fraction=0.25,
):
    """Triangulate finite raw DSM samples that fall inside one footprint."""

    ring_xy = _ring_xy(building, origin_x, origin_y)
    if len(ring_xy) < 3:
        return None, {"roof_surface_source": "unavailable", "roof_valid_fraction": 0.0}
    if source_gsd_m is not None and float(source_gsd_m) > 5.0:
        return None, {"roof_surface_source": "unavailable:coarse-imagery", "roof_valid_fraction": 0.0}

    data = raw_dsm.astype(np.float32).filled(np.nan)
    finite = np.isfinite(data)
    if not finite.any():
        return None, {"roof_surface_source": "unavailable:no-dsm", "roof_valid_fraction": 0.0}

    projected_ring = [
        [x + origin_x, y + origin_y]
        for x, y in ring_xy
    ]
    if projected_ring[0] != projected_ring[-1]:
        projected_ring.append(projected_ring[0])
    try:
        inside = geometry_mask(
            [{"type": "Polygon", "coordinates": [projected_ring]}],
            out_shape=data.shape,
            transform=transform,
            invert=True,
            all_touched=True,
        )
    except (TypeError, ValueError):
        return None, {"roof_surface_source": "unavailable:invalid-footprint", "roof_valid_fraction": 0.0}

    inside_count = int(inside.sum())
    valid_inside = inside & finite
    valid_count = int(valid_inside.sum())
    valid_fraction = valid_count / max(inside_count, 1)
    if valid_count < 3 or valid_fraction < min_valid_fraction:
        return None, {
            "roof_surface_source": "unavailable:insufficient-dsm",
            "roof_valid_fraction": round(valid_fraction, 4),
        }

    sampled = data[valid_inside]
    relief = float(np.percentile(sampled, 95) - np.percentile(sampled, 5))
    # A uniformly flat DSM does not contain enough evidence to replace an
    # explicit OSM/temple height with a half-metre slab. Keep the bounded
    # profile in that case and label it approximate; genuinely varying DSM
    # relief takes the measured path below.
    if relief < 0.5 and float(building.get("height", 0.0) or 0.0) > 2.5:
        return None, {
            "roof_surface_source": "unavailable:flat-dsm",
            "roof_valid_fraction": round(valid_fraction, 4),
            "geometry_quality": "approximate",
            "profile_source": "bounded-procedural-profile",
        }

    rows, cols = np.where(valid_inside)
    height, width = data.shape
    vertex_ids = np.full((height, width), -1, dtype=np.int64)
    vertices = []
    ground_local = float(building.get("ground_elevation", vertical_origin)) - vertical_origin
    try:
        roof_floor = ground_local + 0.3
        roof_cap = ground_local + 50.0
        for index, (row, col) in enumerate(zip(rows, cols)):
            x = transform.c + (float(col) + 0.5) * transform.a + (float(row) + 0.5) * transform.b
            y = transform.f + (float(col) + 0.5) * transform.d + (float(row) + 0.5) * transform.e
            z = float(data[row, col]) - vertical_origin
            vertices.append((x - origin_x, y - origin_y, np.clip(z, roof_floor, roof_cap)))
            vertex_ids[row, col] = index
    except (TypeError, ValueError, OverflowError):
        return None, {
            "roof_surface_source": "unavailable:nonfinite-transform",
            "roof_valid_fraction": round(valid_fraction, 4),
        }

    cell_valid = (
        valid_inside[:-1, :-1]
        & valid_inside[:-1, 1:]
        & valid_inside[1:, :-1]
        & valid_inside[1:, 1:]
    )
    faces = []
    for row, col in zip(*np.where(cell_valid)):
        a = int(vertex_ids[row, col])
        b = int(vertex_ids[row, col + 1])
        c = int(vertex_ids[row + 1, col])
        d = int(vertex_ids[row + 1, col + 1])
        if min(a, b, c, d) >= 0:
            faces.extend(((a, c, b), (b, c, d)))
    if not faces:
        return None, {
            "roof_surface_source": "unavailable:no-contiguous-dsm",
            "roof_valid_fraction": round(valid_fraction, 4),
        }

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    if mesh.is_empty or not np.isfinite(mesh.vertices).all():
        return None, {
            "roof_surface_source": "unavailable:invalid-mesh",
            "roof_valid_fraction": round(valid_fraction, 4),
        }
    return mesh, {
        "roof_surface_source": "metric-dsm-sampled",
        "roof_valid_fraction": round(valid_fraction, 4),
        "geometry_quality": "measured",
        "profile_source": "raw-calibrated-dsm",
    }


def _infer_roof_profile(building, roof_surface):
    """Infer a detailed profile only when measured roof relief supports it."""
    if roof_surface is None or roof_surface.is_empty or len(roof_surface.vertices) < 4:
        return "flat", {"supported": False, "reason": "no-roof-surface"}
    z_values = roof_surface.vertices[:, 2]
    relief = float(np.percentile(z_values, 95) - np.percentile(z_values, 5))
    x_extent = float(np.ptp(roof_surface.vertices[:, 0]))
    y_extent = float(np.ptp(roof_surface.vertices[:, 1]))
    evidence = {
        "supported": relief >= 0.8,
        "relief_m": round(relief, 3),
        "axis": "x" if x_extent >= y_extent else "y",
    }
    if relief < 0.8:
        return "flat", evidence
    # A long footprint with a coherent directional roof rise reads as gable;
    # compact roofs with multi-directional relief use a conservative hip cap.
    profile = "gable" if max(x_extent, y_extent) / max(min(x_extent, y_extent), 0.5) >= 1.35 else "hip"
    evidence["profile"] = profile
    return profile, evidence


def _building_mesh(building, origin_x, origin_y, vertical_origin, roof_elevation=None):
    polygon = building.get("polygon_projected") or []
    if len(polygon) < 3:
        return None
    ground = float(building.get("ground_elevation", vertical_origin)) - vertical_origin + 0.05
    roof_value = roof_elevation
    if roof_value is None:
        roof_value = float(building.get("roof_elevation", building.get("ground_elevation", vertical_origin)))
    roof = float(roof_value) - vertical_origin
    if not np.isfinite(ground) or not np.isfinite(roof) or roof <= ground:
        return None
    ring = _ring_xy(building, origin_x, origin_y)
    if len(ring) < 3:
        return None
    # OSM ways are not guaranteed to use one winding direction. Normalize the
    # ring so the roof is consistently upward and the floor consistently
    # downward; this keeps WebGL back-face culling from making some buildings
    # appear hollow or invisible.
    signed_area = sum(
        ring[index][0] * ring[(index + 1) % len(ring)][1]
        - ring[(index + 1) % len(ring)][0] * ring[index][1]
        for index in range(len(ring))
    ) * 0.5
    if signed_area < 0:
        ring.reverse()

    vertices = np.asarray(
        [(x, y, ground) for x, y in ring] + [(x, y, roof) for x, y in ring],
        dtype=np.float32,
    )
    count = len(ring)
    triangles = _triangulate_ring(ring)
    if not triangles:
        return None
    faces = [[a, c, b] for a, b, c in triangles]
    faces.extend([[count + a, count + b, count + c] for a, b, c in triangles])
    for i in range(count):
        j = (i + 1) % count
        faces.extend([[i, j, count + i], [j, count + j, count + i]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.visual.material = SimpleMaterial(
        diffuse=(174, 164, 145, 255),
    )
    return mesh


def _building_roof_mesh(building, origin_x, origin_y, vertical_origin, roof_surface=None):
    polygon = building.get("polygon_projected") or []
    if len(polygon) < 3:
        return None
    roof = float(building.get("roof_elevation", vertical_origin)) - vertical_origin + 0.08
    ring_xy = _ring_xy(building, origin_x, origin_y)
    if len(ring_xy) < 3:
        return None
    # The raw DSM surface is used to measure the roof height and infer a
    # supported architectural profile, but it is not emitted directly: a
    # sparse raster footprint can produce an open surface with holes. The
    # profile builder below always returns a closed roof shell instead.
    if roof_surface is not None and not roof_surface.is_empty:
        measured_top = float(np.max(roof_surface.vertices[:, 2])) + 0.08
        if np.isfinite(measured_top):
            roof = measured_top

    def sealed_flat_slab():
        """Use the accepted wall footprint for a guaranteed closed roof cap."""
        thickness = _roof_shell_thickness(ring_xy)
        roof_absolute = float(building.get("roof_elevation", vertical_origin))
        slab = {
            **building,
            # _building_mesh adds its normal wall clearance to the base. Keep
            # this cap just above the wall mass without leaving a visible gap.
            "ground_elevation": roof_absolute - thickness - 0.05,
            "roof_elevation": roof_absolute,
        }
        return _building_mesh(
            slab,
            origin_x,
            origin_y,
            vertical_origin,
            roof_elevation=roof_absolute,
        )

    profile = _profile_roof_mesh(building, ring_xy, roof)
    if profile is not None and profile.is_watertight:
        return profile
    # A malformed architectural profile must still leave the building with a
    # sealed terrace. The flat fallback is deliberately tiny and sits above
    # the closed wall massing without z-fighting.
    fallback = {**building, "geometry_profile": "flat"}
    profile = _profile_roof_mesh(fallback, ring_xy, roof)
    if profile is not None and profile.is_watertight:
        return profile
    slab = sealed_flat_slab()
    if slab is not None and not slab.is_watertight:
        trimesh_repair.fill_holes(slab)
        if hasattr(slab, "remove_unreferenced_vertices"):
            slab.remove_unreferenced_vertices()
    if slab is not None:
        slab.visual.material = SimpleMaterial(diffuse=(194, 168, 117, 255))
    return slab


def _sloped_roof_mesh(building, origin_x, origin_y, vertical_origin, roof_surface=None):
    """Create a sloped roof mesh when DSM data shows significant relief."""
    polygon = building.get("polygon_projected") or []
    if len(polygon) < 3:
        return None
    ring_xy = _ring_xy(building, origin_x, origin_y)
    if len(ring_xy) < 3:
        return None

    if roof_surface is None or roof_surface.is_empty:
        return None

    # Use the DSM-sampled roof surface but add a ridge line for more
    # realistic roof geometry when there's enough relief
    measured = roof_surface.copy()
    roof_verts = measured.vertices
    if len(roof_verts) < 4:
        return None

    # Check if there's enough relief to justify a sloped roof
    z_values = roof_verts[:, 2]
    relief = float(np.percentile(z_values, 95) - np.percentile(z_values, 5))
    if relief < 0.5:
        return None

    # Add a ridge line along the longest axis for a gabled roof effect
    x_range = float(np.max(roof_verts[:, 0]) - np.min(roof_verts[:, 0]))
    y_range = float(np.max(roof_verts[:, 1]) - np.min(roof_verts[:, 1]))

    if x_range > y_range:
        # Ridge along X axis
        y_center = float(np.mean(roof_verts[:, 1]))
        ridge_z = float(np.percentile(z_values, 90)) + 0.3
        ridge_points = [
            (float(np.min(roof_verts[:, 0])), y_center, ridge_z),
            (float(np.max(roof_verts[:, 0])), y_center, ridge_z),
        ]
    else:
        # Ridge along Y axis
        x_center = float(np.mean(roof_verts[:, 0]))
        ridge_z = float(np.percentile(z_values, 90)) + 0.3
        ridge_points = [
            (x_center, float(np.min(roof_verts[:, 1])), ridge_z),
            (x_center, float(np.max(roof_verts[:, 1])), ridge_z),
        ]

    # Add ridge vertices and connect to existing roof
    ridge_start = len(roof_verts)
    all_verts = np.vstack((roof_verts, np.asarray(ridge_points, dtype=np.float32)))

    # Find edge vertices and connect to ridge
    faces = list(measured.faces)
    edge_indices = np.where(np.bincount(measured.faces.ravel(), minlength=len(roof_verts)) == 1)[0]
    for idx in edge_indices:
        faces.append([int(idx), ridge_start, ridge_start + 1])

    mesh = trimesh.Trimesh(vertices=all_verts, faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.visual.material = SimpleMaterial(diffuse=(194, 168, 117, 255))
    return mesh


def _temple_roof_mesh(building, ring_xy, roof):
    """Create a bounded, footprint-conforming temple roof/spire profile."""
    if len(ring_xy) < 3:
        return None
    points = np.asarray(ring_xy, dtype=np.float32)
    centroid = points.mean(axis=0)
    extents = points.max(axis=0) - points.min(axis=0)
    short_side = float(max(0.5, min(extents[0], extents[1])))
    base_height = max(
        1.0,
        float(building.get("height") or (
            float(building.get("roof_elevation", roof))
            - float(building.get("ground_elevation", roof))
        )),
    )
    # The spire is deliberately conservative: it adds readable temple
    # character without inventing a landmark-scale tower from a nadir image.
    spire_height = min(max(2.0, base_height * 1.2), max(3.0, short_side * 0.25))

    def scaled(factor, z):
        ring = centroid + (points - centroid) * factor
        return np.column_stack((ring, np.full(len(ring), z, dtype=np.float32)))

    lower = scaled(0.58, roof + 0.08)
    middle = scaled(0.42, roof + spire_height * 0.42)
    upper = scaled(0.23, roof + spire_height * 0.78)
    apex = np.asarray([[centroid[0], centroid[1], roof + spire_height]], dtype=np.float32)
    vertices = np.vstack((lower, middle, upper, apex))
    n = len(points)
    faces = []

    def frustum(start, end):
        for index in range(n):
            nxt = (index + 1) % n
            faces.extend([
                [start + index, start + nxt, end + index],
                [start + nxt, end + nxt, end + index],
            ])

    frustum(0, n)
    frustum(n, 2 * n)
    for index in range(n):
        nxt = (index + 1) % n
        faces.append([2 * n + index, 2 * n + nxt, 3 * n])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)


def _polygon_mesh(feature, origin_x, origin_y, vertical_origin, color, z_offset=0.05):
    polygon = feature.get("polygon_projected") or []
    if len(polygon) < 3:
        return None
    ground = float(feature.get("ground_elevation", vertical_origin)) - vertical_origin + z_offset
    ring = [(float(point[0]) - origin_x, float(point[1]) - origin_y, ground) for point in polygon]
    vertices = np.asarray(ring, dtype=np.float32)
    triangles = _triangulate_ring([(x, y) for x, y, _ in ring])
    if not triangles:
        return None
    faces = np.asarray([[a, b, c] for a, b, c in triangles], dtype=np.int64)
    if not len(faces):
        return None
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.material = SimpleMaterial(diffuse=color)
    return mesh


def _road_mesh(road, origin_x, origin_y, vertical_origin):
    path = road.get("path_projected") or []
    if len(path) < 2:
        return None
    width = max(0.8, float(road.get("width_m", 4.0)))
    ground = float(road.get("ground_elevation", vertical_origin)) - vertical_origin + 0.08
    vertices = []
    for index, point in enumerate(path):
        previous = path[max(0, index - 1)]
        following = path[min(len(path) - 1, index + 1)]
        dx = float(following[0] - previous[0])
        dy = float(following[1] - previous[1])
        length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        nx, ny = -dy / length * width * 0.5, dx / length * width * 0.5
        x = float(point[0]) - origin_x
        y = float(point[1]) - origin_y
        vertices.extend([(x - nx, y - ny, ground), (x + nx, y + ny, ground)])
    faces = []
    for index in range(len(path) - 1):
        left, right = index * 2, index * 2 + 1
        next_left, next_right = (index + 1) * 2, (index + 1) * 2 + 1
        faces.extend([[left, next_left, right], [right, next_left, next_right]])
    if not faces:
        return None
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float32), faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.visual.material = SimpleMaterial(diffuse=(58, 58, 54, 255))
    return mesh


def _semantic_wire_mesh(region, origin_x, origin_y, vertical_origin):
    """Create a thin ground-level ribbon around one semantic region."""
    polygon = region.get("polygon_projected") or []
    if len(polygon) < 3:
        return None
    path = polygon + [polygon[0]]
    mesh = _road_mesh(
        {
            "path_projected": path,
            "width_m": max(0.18, min(1.2, float(region.get("line_width_m", 0.35)))),
            "ground_elevation": region.get("ground_elevation", vertical_origin),
        },
        origin_x,
        origin_y,
        vertical_origin,
    )
    if mesh is not None:
        colors = {
            "building": (230, 70, 60, 255),
            "road": (90, 90, 95, 255),
            "vegetation": (60, 160, 70, 255),
            "water": (60, 120, 230, 255),
            "bare_ground": (190, 170, 120, 255),
            "infrastructure": (200, 130, 40, 255),
        }
        mesh.visual.material = SimpleMaterial(diffuse=colors.get(region.get("class"), (255, 220, 80, 255)))
    return mesh


def _tree_mesh(tree, origin_x, origin_y, vertical_origin):
    ground = float(tree.get("ground_elevation", vertical_origin)) - vertical_origin
    height = max(2.5, float(tree.get("height", 8.0)))
    radius = max(1.2, float(tree.get("canopy_radius", height * 0.32)))
    point = tree.get("point_projected") or []
    if len(point) != 2:
        return None
    x, y = float(point[0]) - origin_x, float(point[1]) - origin_y
    trunk = trimesh.creation.cylinder(radius=max(0.15, radius * 0.12), height=height * 0.42, sections=7)
    trunk.apply_translation((x, y, ground + height * 0.21))
    canopy = trimesh.creation.cone(radius=radius, height=height * 0.7, sections=8)
    canopy.apply_translation((x, y, ground + height * 0.65))
    mesh = trimesh.util.concatenate([trunk, canopy])
    mesh.visual.material = SimpleMaterial(diffuse=(55, 112, 61, 255))
    return mesh


def _closed_roof_shell(top_vertices, top_faces, base_ring, base_z):
    """Close a roof surface with eave walls and a bottom cap."""
    if len(base_ring) < 3 or not top_faces:
        return None
    base_ring = np.asarray(base_ring, dtype=np.float32)
    top_vertices = np.asarray(top_vertices, dtype=np.float32)
    base_vertices = np.column_stack((base_ring, np.full(len(base_ring), float(base_z), dtype=np.float32)))
    vertices = np.vstack((top_vertices, base_vertices))
    base_start = len(top_vertices)
    faces = [list(face) for face in top_faces]
    for index in range(len(base_ring)):
        nxt = (index + 1) % len(base_ring)
        faces.extend((
            [index, base_start + index, base_start + nxt],
            [index, base_start + nxt, nxt],
        ))
    for a, b, c in _triangulate_ring(base_ring.tolist()):
        faces.append([base_start + a, base_start + c, base_start + b])
    if not faces:
        return None
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)
    if mesh.is_empty or not np.isfinite(mesh.vertices).all():
        return None
    if not mesh.is_watertight:
        trimesh_repair.fill_holes(mesh)
        if hasattr(mesh, "remove_unreferenced_vertices"):
            mesh.remove_unreferenced_vertices()
    mesh.visual.material = SimpleMaterial(diffuse=(194, 168, 117, 255))
    return mesh


def _roof_shell_thickness(ring_xy):
    points = np.asarray(ring_xy, dtype=np.float32)
    if points.ndim != 2 or len(points) < 3:
        return 0.12
    extents = np.maximum(points.max(axis=0) - points.min(axis=0), 0.5)
    return max(0.08, min(0.3, float(min(extents)) * 0.025))


def _profile_roof_mesh(building, ring_xy, roof):
    """Create a bounded, watertight architectural roof shell."""
    if len(ring_xy) < 3:
        return None
    profile = str(building.get("geometry_profile") or "flat").lower()
    points = np.asarray(ring_xy, dtype=np.float32)
    centroid = points.mean(axis=0)
    if not _point_in_ring((float(centroid[0]), float(centroid[1])), ring_xy):
        return None
    # Keep all generated eaves comfortably inside the accepted footprint.
    eaves = centroid + (points - centroid) * 0.88
    if not all(_point_in_ring((float(point[0]), float(point[1])), ring_xy) for point in eaves):
        return None
    extents = np.maximum(points.max(axis=0) - points.min(axis=0), 0.5)
    rise = min(max(0.8, float(building.get("height", 3.0)) * 0.18), max(1.0, float(min(extents)) * 0.45))
    thickness = _roof_shell_thickness(ring_xy)
    base_z = float(roof) - thickness
    eave_z = float(roof)

    if profile in {"temple", "towered"}:
        lower = centroid + (points - centroid) * 0.58
        middle = centroid + (points - centroid) * 0.42
        upper = centroid + (points - centroid) * 0.23
        spire_height = min(max(2.0, float(building.get("height", 3.0)) * 1.2), max(3.0, float(min(extents)) * 0.25))
        vertices = np.vstack((
            np.column_stack((lower, np.full(len(lower), eave_z + 0.08, dtype=np.float32))),
            np.column_stack((middle, np.full(len(middle), eave_z + spire_height * 0.42, dtype=np.float32))),
            np.column_stack((upper, np.full(len(upper), eave_z + spire_height * 0.78, dtype=np.float32))),
            np.asarray([[centroid[0], centroid[1], eave_z + spire_height]], dtype=np.float32),
        ))
        n = len(points)
        faces = []
        for start, end in ((0, n), (n, 2 * n)):
            for index in range(n):
                nxt = (index + 1) % n
                faces.extend(([start + index, start + nxt, end + index], [start + nxt, end + nxt, end + index]))
        apex = 3 * n
        for index in range(n):
            nxt = (index + 1) % n
            faces.append([2 * n + index, 2 * n + nxt, apex])
        mesh = _closed_roof_shell(vertices, faces, eaves, base_z)
        if mesh is not None:
            building["roof_plane_count"] = 4
        return mesh

    top_eaves = np.column_stack((eaves, np.full(len(eaves), eave_z, dtype=np.float32)))
    if profile == "hip":
        apex = np.asarray([[centroid[0], centroid[1], eave_z + rise]], dtype=np.float32)
        vertices = np.vstack((top_eaves, apex))
        apex_index = len(eaves)
        faces = [[index, (index + 1) % len(eaves), apex_index] for index in range(len(eaves))]
        building["roof_plane_count"] = 4
        return _closed_roof_shell(vertices, faces, eaves, base_z)

    if profile in {"gable", "compound"}:
        # The bounded gable construction below is exact for a four-corner
        # footprint.  Irregular/concave footprints use a sealed hip fallback
        # instead of risking a ridge that crosses a courtyard or leaves an
        # open roof boundary.
        if len(eaves) != 4:
            fallback = {**building, "geometry_profile": "hip"}
            return _profile_roof_mesh(fallback, ring_xy, roof)
        axis = np.asarray([1.0, 0.0], dtype=np.float32) if extents[0] >= extents[1] else np.asarray([0.0, 1.0], dtype=np.float32)
        half = float(max(extents) * 0.30)
        ridge = np.asarray([centroid - axis * half, centroid + axis * half], dtype=np.float32)
        # A concave footprint can make a naive ridge bridge its courtyard.
        # Fall back to a sealed flat cap rather than emitting geometry outside
        # the accepted building footprint.
        if not all(_point_in_ring((float(point[0]), float(point[1])), ring_xy) for point in ridge):
            fallback = {**building, "geometry_profile": "flat"}
            return _profile_roof_mesh(fallback, ring_xy, roof)
        ridge_z = np.full((2, 1), eave_z + rise, dtype=np.float32)
        vertices = np.vstack((top_eaves, np.hstack((ridge, ridge_z))))
        faces = []
        for index in range(len(eaves)):
            nxt = (index + 1) % len(eaves)
            midpoint = (eaves[index] + eaves[nxt]) * 0.5
            along_axis = float(np.dot(midpoint - centroid, axis))
            if abs(along_axis) <= max(float(min(extents)) * 0.08, 1e-5):
                # Long eave: bridge both ridge endpoints with a two-triangle
                # roof plane. The two end edges below cap the gable ends.
                start_ridge = len(eaves) if float(np.dot(eaves[index] - centroid, axis)) < 0 else len(eaves) + 1
                end_ridge = len(eaves) if float(np.dot(eaves[nxt] - centroid, axis)) < 0 else len(eaves) + 1
                faces.extend(([index, nxt, start_ridge], [nxt, end_ridge, start_ridge]))
            else:
                ridge_index = len(eaves) if along_axis < 0 else len(eaves) + 1
                faces.append([index, nxt, ridge_index])
        building["roof_plane_count"] = 2 if profile == "gable" else 4
        return _closed_roof_shell(vertices, faces, eaves, base_z)

    # Flat and unknown profiles get a thin sealed slab, preventing open
    # terraces while preserving the measured wall height.
    triangles = _triangulate_ring(eaves.tolist())
    return _closed_roof_shell(top_eaves, triangles, eaves, base_z)


def _point_in_ring(point, ring):
    if len(ring) < 3:
        return False
    x, y = point
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = current
        x2, y2 = previous
        if (y1 > y) != (y2 > y) and abs(y2 - y1) > 1e-12:
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
        previous = current
    return inside


def _segments_intersect(a, b, c, d):
    def orientation(p, q, r):
        value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(value) < 1e-9:
            return 0
        return 1 if value > 0 else 2

    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    return o1 != o2 and o3 != o4


def _rings_intersect(first, second):
    if len(first) < 3 or len(second) < 3:
        return False
    if _point_in_ring(first[0], second) or _point_in_ring(second[0], first):
        return True
    for index, start in enumerate(first):
        end = first[(index + 1) % len(first)]
        for other_index, other_start in enumerate(second):
            other_end = second[(other_index + 1) % len(second)]
            if _segments_intersect(start, end, other_start, other_end):
                return True
    return False


def _feature_hits_building(feature, buildings, kind):
    """Reject context geometry that would be rendered over a roof."""
    if kind not in {"landcover", "trees", "water", "roads"}:
        return False
    polygon = feature.get("polygon_projected") or []
    point = feature.get("point_projected") or []
    samples = []
    if len(point) == 2:
        try:
            samples.append((float(point[0]), float(point[1])))
            radius = max(0.5, float(feature.get("canopy_radius", 2.0)))
            for angle in np.linspace(0.0, 2.0 * np.pi, 9)[:-1]:
                samples.append((float(point[0]) + radius * np.cos(angle), float(point[1]) + radius * np.sin(angle)))
        except (TypeError, ValueError):
            pass
    if len(polygon) >= 3:
        samples.extend((float(p[0]), float(p[1])) for p in polygon)
        samples.append((
            sum(float(p[0]) for p in polygon) / len(polygon),
            sum(float(p[1]) for p in polygon) / len(polygon),
        ))
    for building in buildings or []:
        footprint = building.get("polygon_projected") or []
        if len(footprint) < 3:
            continue
        if len(polygon) >= 3 and _rings_intersect(polygon, footprint):
            return True
        if any(_point_in_ring(sample, footprint) for sample in samples):
            return True
        # A vegetation polygon can surround a footprint without having any
        # of its own vertices inside it.  The reverse centroid check handles
        # that common land-cover case without a heavyweight geometry library.
        if len(polygon) >= 3:
            center = (
                sum(float(p[0]) for p in footprint) / len(footprint),
                sum(float(p[1]) for p in footprint) / len(footprint),
            )
            if _point_in_ring(center, polygon):
                return True
    if kind == "roads":
        path = feature.get("path_projected") or []
        for building in buildings or []:
            footprint = building.get("polygon_projected") or []
            if len(footprint) < 3 or len(path) < 2:
                continue
            if any(_point_in_ring((float(point[0]), float(point[1])), footprint) for point in path):
                return True
            for index in range(len(path) - 1):
                start, end = path[index], path[index + 1]
                for edge_index, edge_start in enumerate(footprint):
                    edge_end = footprint[(edge_index + 1) % len(footprint)]
                    if _segments_intersect(start, end, edge_start, edge_end):
                        return True
    return False


def _safe_feature_id(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(fallback)


def _scene_with_buildings(
    terrain,
    metadata,
    buildings,
    environment=None,
    raw_dsm=None,
    raw_transform=None,
    source_gsd_m=None,
    texture_image=None,
    texture_bounds=None,
):
    scene = trimesh.Scene()
    scene.add_geometry(terrain, node_name="TERRAIN", geom_name="TERRAIN")
    added = 0
    building_vertices = 0
    building_faces = 0
    building_geometry_qa = {
        "building_meshes_checked": 0,
        "building_meshes_watertight": 0,
        "building_roofs_watertight": 0,
        "building_meshes_repaired": 0,
        "building_meshes_rejected": 0,
        "building_roofs_rejected": 0,
    }
    for building in buildings or []:
        roof_surface = None
        trusted_height = str(building.get("height_source", "")).lower().startswith(("osm:height", "osm:levels"))
        roof_stats = {
            "roof_surface_source": "unavailable:no-raw-dsm",
            "roof_valid_fraction": 0.0,
            "geometry_quality": "measured" if trusted_height else "approximate",
            "profile_source": "osm-explicit-height" if trusted_height else "bounded-procedural-profile",
        }
        if trusted_height:
            roof_stats["roof_surface_source"] = "osm-explicit-height"
        if raw_dsm is not None and raw_transform is not None and not trusted_height:
            roof_surface, sampled_stats = _roof_surface_mesh(
                building,
                raw_dsm,
                raw_transform,
                metadata["origin_x"],
                metadata["origin_y"],
                metadata["vertical_origin_m"],
                source_gsd_m=source_gsd_m,
            )
            roof_stats.update(sampled_stats)
        building.update(roof_stats)
        explicit_roof_shape = str((building.get("osm_tags") or {}).get("roof:shape") or "").lower()
        if roof_surface is not None and str(building.get("geometry_profile") or "flat") in {"flat", "standard"} and explicit_roof_shape not in {"flat", "flat_roof"}:
            inferred_profile, roof_evidence = _infer_roof_profile(building, roof_surface)
            building["roof_evidence"] = {**(building.get("roof_evidence") or {}), **roof_evidence}
            if roof_evidence.get("supported"):
                building["geometry_profile"] = inferred_profile
                building["roof_type"] = inferred_profile
                building["roof_plane_count"] = {"gable": 2, "hip": 4}.get(inferred_profile, 1)
                building["profile_source"] = "metric-dsm-roof-relief"
        if roof_surface is not None and not roof_surface.is_empty:
            measured_top = float(np.max(roof_surface.vertices[:, 2])) + metadata["vertical_origin_m"]
            if np.isfinite(measured_top):
                building["roof_elevation"] = measured_top
                ground_value = float(building.get("ground_elevation", metadata["vertical_origin_m"]))
                if np.isfinite(ground_value):
                    building["height"] = round(max(0.5, measured_top - ground_value), 3)
                    building["height_source"] = "metric-dsm-sampled"
        try:
            ring_xy = _ring_xy(building, metadata["origin_x"], metadata["origin_y"])
            roof_value = float(building.get("roof_elevation", metadata["vertical_origin_m"]))
            wall_roof_value = roof_value - _roof_shell_thickness(ring_xy) - 0.02
            mesh = _building_mesh(
                building,
                metadata["origin_x"],
                metadata["origin_y"],
                metadata["vertical_origin_m"],
                roof_elevation=wall_roof_value,
            )
        except (TypeError, ValueError, IndexError, OverflowError):
            building_geometry_qa["building_meshes_rejected"] += 1
            building["geometry_quality"] = "rejected-invalid-values"
            continue
        if mesh is None:
            building_geometry_qa["building_meshes_rejected"] += 1
            building["geometry_quality"] = "rejected-invalid-footprint"
            continue
        building_geometry_qa["building_meshes_checked"] += 1
        if not mesh.is_watertight:
            trimesh_repair.fill_holes(mesh)
            if hasattr(mesh, "remove_unreferenced_vertices"):
                mesh.remove_unreferenced_vertices()
            if mesh.is_watertight:
                building_geometry_qa["building_meshes_repaired"] += 1
        if not mesh.is_watertight:
            building_geometry_qa["building_meshes_rejected"] += 1
            building["geometry_quality"] = "rejected-nonwatertight"
            continue
        building_geometry_qa["building_meshes_watertight"] += 1
        identifier = _safe_feature_id(building.get("id"), added + 1)
        scene.add_geometry(mesh, node_name=f"BUILDING_{identifier}_{added + 1}", geom_name=f"BUILDING_{identifier}_{added + 1}")
        roof_error = False
        try:
            roof = _building_roof_mesh(
                building,
                metadata["origin_x"],
                metadata["origin_y"],
                metadata["vertical_origin_m"],
                roof_surface=roof_surface,
            )
        except (TypeError, ValueError, IndexError, OverflowError):
            roof = None
            roof_error = True
            building_geometry_qa["building_roofs_rejected"] += 1
        if roof is not None and roof.is_watertight:
            if texture_image is not None:
                apply_rgb_texture(roof, texture_image, *(texture_bounds or (None, None)))
            scene.add_geometry(roof, node_name=f"ROOF_BUILDING_{identifier}_{added + 1}", geom_name=f"ROOF_BUILDING_{identifier}_{added + 1}")
            building_geometry_qa["building_roofs_watertight"] += 1
        elif roof is not None:
            building_geometry_qa["building_roofs_rejected"] += 1
        elif not roof_error:
            building_geometry_qa["building_roofs_rejected"] += 1
        added += 1
        building_vertices += len(mesh.vertices)
        building_faces += len(mesh.faces)
    environment = environment or {}
    environment_counts = {
        "roads": 0,
        "water": 0,
        "landcover": 0,
        "trees": 0,
        "semantic": 0,
        "geometry_qa": {
            **building_geometry_qa,
            "vegetation_building_intersections": 0,
            "road_building_intersections": 0,
            "water_building_intersections": 0,
            "context_meshes_rejected": 0,
            "emitted_context_intersections": 0,
        },
        "semantic_layers": {
            "building": 0,
            "vegetation": 0,
            "water": 0,
            "road": 0,
            "bare_ground": 0,
            "infrastructure": 0,
        },
    }
    colors = {
        "water": (48, 126, 160, 220),
        "forest": (53, 112, 61, 210),
        "park": (92, 136, 70, 190),
    }
    for feature in environment.get("roads") or []:
        if _feature_hits_building(feature, buildings, "roads"):
            environment_counts["geometry_qa"]["road_building_intersections"] += 1
            environment_counts["geometry_qa"]["context_meshes_rejected"] += 1
            continue
        mesh = _road_mesh(feature, metadata["origin_x"], metadata["origin_y"], metadata["vertical_origin_m"])
        if mesh is not None:
            name = f"ROAD_{_safe_feature_id(feature.get('id'), environment_counts['roads'] + 1)}_{environment_counts['roads'] + 1}"
            scene.add_geometry(mesh, node_name=name, geom_name=name)
            environment_counts["roads"] += 1
    # Context layers are deliberately ground-hugging.  They are isolated from
    # the bare terrain and rejected where they intersect an accepted building,
    # so a canopy or semantic region cannot cut through a roof.
    for layer in ("water", "landcover"):
        for feature in environment.get(layer) or []:
            if _feature_hits_building(feature, buildings, layer):
                qa_key = "water_building_intersections" if layer == "water" else "vegetation_building_intersections"
                environment_counts["geometry_qa"][qa_key] += 1
                environment_counts["geometry_qa"]["context_meshes_rejected"] += 1
                continue
            mesh = _polygon_mesh(
                feature,
                metadata["origin_x"],
                metadata["origin_y"],
                metadata["vertical_origin_m"],
                colors.get("water" if layer == "water" else feature.get("class", "park"), colors["park"]),
                z_offset=0.12 if layer == "water" else 0.06,
            )
            if mesh is not None:
                name = f"{'WATER' if layer == 'water' else 'VEGETATION'}_{_safe_feature_id(feature.get('id'), environment_counts[layer] + 1)}_{environment_counts[layer] + 1}"
                scene.add_geometry(mesh, node_name=name, geom_name=name)
                environment_counts[layer] += 1
    for tree in environment.get("trees") or []:
        if _feature_hits_building(tree, buildings, "trees"):
            environment_counts["geometry_qa"]["vegetation_building_intersections"] += 1
            environment_counts["geometry_qa"]["context_meshes_rejected"] += 1
            continue
        mesh = _tree_mesh(tree, metadata["origin_x"], metadata["origin_y"], metadata["vertical_origin_m"])
        if mesh is not None:
            name = f"VEGETATION_TREE_{_safe_feature_id(tree.get('id'), environment_counts['trees'] + 1)}_{environment_counts['trees'] + 1}"
            scene.add_geometry(mesh, node_name=name, geom_name=name)
            environment_counts["trees"] += 1
    for region in (environment.get("semantic_regions") or [])[:300]:
        if _feature_hits_building(region, buildings, "landcover"):
            environment_counts["geometry_qa"]["vegetation_building_intersections"] += 1
            environment_counts["geometry_qa"]["context_meshes_rejected"] += 1
            continue
        mesh = _semantic_wire_mesh(region, metadata["origin_x"], metadata["origin_y"], metadata["vertical_origin_m"])
        if mesh is not None:
            name = f"SEMANTIC_WIREFRAME_{region.get('class', 'other').upper()}_{environment_counts['semantic'] + 1}"
            scene.add_geometry(mesh, node_name=name, geom_name=name)
            environment_counts["semantic"] += 1
            semantic_class = str(region.get("class", "infrastructure"))
            environment_counts["semantic_layers"][semantic_class] = environment_counts["semantic_layers"].get(semantic_class, 0) + 1
    return scene, added, building_vertices, building_faces, environment_counts


def _terrain_overrides(buildings=None, environment=None):
    """Collect polygons that should sit on top of bare terrain, not inside it."""
    environment = environment or {}
    overrides = []
    for layer in ("landcover", "roads", "water", "exclusion_zones"):
        for feature in environment.get(layer) or []:
            if feature.get("polygon_projected"):
                overrides.append(feature)
            elif layer == "roads" and feature.get("path_projected"):
                overrides.append({**feature, "polygon_projected": feature.get("path_projected")})
    # Semantic regions are evidence/diagnostics only. Flattening every raw
    # segmentation ribbon turns vegetation and uncertain roofs into a large
    # opaque hill. Only accepted physical layers alter the terrain surface.
    # Buildings are applied last so their isolated walls/roofs sit on bare
    # ground even when a mapped environment polygon touches them.
    overrides.extend(item for item in (buildings or []) if item.get("polygon_projected"))
    return overrides


def run_reconstruction(
    dsm_path,
    rgb_path,
    output_path,
    size=512,
    metadata_path=None,
    progress_callback=None,
    z_units="meters",
    buildings=None,
    environment=None,
    source_gsd_m=None,
    terrain_surface_path=None,
):
    """Create a textured GLB from the supplied DSM and RGB raster."""
    dsm_path = Path(dsm_path)
    rgb_path = Path(rgb_path)
    output_path = Path(output_path)
    metadata_path = Path(metadata_path) if metadata_path else output_path.with_suffix(".json")
    if not dsm_path.exists():
        raise FileNotFoundError(f"DSM not found: {dsm_path}")
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB raster not found: {rgb_path}")
    if progress_callback:
        progress_callback({"fraction": 0.15, "sub": "creating mesh from DSM"})
    terrain_surface_path = Path(terrain_surface_path) if terrain_surface_path else None
    terrain_source = terrain_surface_path if terrain_surface_path and terrain_surface_path.exists() else dsm_path
    terrain_overrides = [] if terrain_source != dsm_path else _terrain_overrides(buildings, environment)
    # Keep this unmodified calibrated grid for measured roof sampling. The
    # terrain call below receives a separate copy with only accepted physical
    # features flattened beneath their isolated scene layers.
    raw_dsm, raw_transform, _raw_crs, _raw_bounds, _raw_nodata, _raw_overrides, _raw_filtered = load_downsampled_dsm(
        dsm_path,
        target_size=size,
        surface_overrides=None,
    )
    mesh, metadata = create_dsm_mesh(
        terrain_source,
        target_size=size,
        z_units=z_units,
        surface_overrides=terrain_overrides,
    )
    if progress_callback:
        progress_callback({"fraction": 0.55, "sub": "applying RGB texture"})
    texture_image = load_rgb_texture_image(rgb_path, texture_width=size * 2, texture_height=size * 2)
    texture_bounds = (
        (float(np.min(mesh.vertices[:, 0])), float(np.max(mesh.vertices[:, 0]))),
        (float(np.min(mesh.vertices[:, 1])), float(np.max(mesh.vertices[:, 1]))),
    )
    mesh = apply_rgb_texture(mesh, texture_image, *texture_bounds)
    if not mesh.is_empty and (not mesh.vertices.size or not mesh.faces.size):
        raise ValueError("3D reconstruction produced an empty mesh.")
    if not mesh.is_empty:
        if not np.isfinite(mesh.vertices).all():
            raise ValueError("3D reconstruction produced non-finite vertices.")
        uv = getattr(mesh.visual, "uv", None)
        if uv is None or not np.isfinite(uv).all():
            raise ValueError("3D reconstruction did not produce valid texture coordinates.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback({"fraction": 0.72, "sub": "adding buildings and map environment" if (buildings or environment) else "exporting GLB"})
    scene, building_count, building_vertices, building_faces, environment_counts = _scene_with_buildings(
        mesh,
        metadata,
        buildings or [],
        environment,
        raw_dsm=raw_dsm,
        raw_transform=raw_transform,
        source_gsd_m=source_gsd_m,
        texture_image=texture_image,
        texture_bounds=texture_bounds,
    )
    if progress_callback:
        progress_callback({"fraction": 0.82, "sub": "exporting GLB"})
    export_glb(scene, output_path)
    with rasterio.open(dsm_path) as source:
        source_bounds = [float(source.bounds.left), float(source.bounds.bottom), float(source.bounds.right), float(source.bounds.top)]
        source_resolution = [float(np.hypot(source.transform.a, source.transform.d)), float(np.hypot(source.transform.b, source.transform.e))]
        source_grid = {"width": int(source.width), "height": int(source.height), "resolution_m": source_resolution, "bounds": source_bounds}
        source_crs = str(source.crs) if source.crs else None
    mesh_width = int(metadata.get("width", 0))
    mesh_height = int(metadata.get("height", 0))
    scene_alignment = {
        "crs": source_crs or metadata.get("crs"),
        "source_grid": source_grid,
        "mesh_grid": {
            "width": mesh_width,
            "height": mesh_height,
            "resolution_m": [float(metadata.get("resolution_x", 0.0)), float(metadata.get("resolution_y", 0.0))],
            "origin": [float(metadata.get("origin_x", 0.0)), float(metadata.get("origin_y", 0.0))],
            "projected_bounds": list(metadata.get("bounds") or source_bounds),
        },
        "source_resolution_m": source_resolution,
        "projected_bounds": source_bounds,
        "geometry_contract": "projected-feature-coordinates",
    }

    metadata.update({
        "source_dsm": str(dsm_path),
        "terrain_surface_source": str(terrain_source),
        "terrain_surface_provenance": "persisted-visualization-bare-earth" if terrain_source != dsm_path else "downsampled-feature-overrides",
        "source_rgb": str(rgb_path),
        "output_glb": str(output_path),
        "visualization_target_size": size,
        "coordinate_system": "Local Y-up mesh coordinates",
        "z_units": z_units,
        "horizontal_units": "meters" if z_units == "meters" else "relative grid units",
        "vertical_units": "meters" if z_units == "meters" else "relative units",
        "vertical_origin_m": metadata.get("vertical_origin_m"),
        "scene_alignment": scene_alignment,
        "texture_valid": True,
        "mesh_valid": True,
        "description": "ASTERRA DSM converted to a textured 3D mesh. Original geospatial reference is preserved in this metadata.",
        "buildings_detected": building_count,
        "building_vertices": building_vertices,
        "building_faces": building_faces,
        "has_buildings": building_count > 0,
        "environment_counts": environment_counts,
        "geometry_qa": environment_counts.get("geometry_qa", {}),
        "segmentation_wireframes": {
            "count": int(environment_counts.get("semantic", 0)),
            "default_visible": False,
        },
        "has_environment": any(
            int(value) > 0
            for key, value in environment_counts.items()
            if key not in {"semantic", "semantic_layers", "geometry_qa"}
        ),
        "terrain_overrides": len(terrain_overrides),
        "reconstruction": {
            "surface_mode": "hybrid-dsm-semantic",
            "roof_measured": sum(
                1 for building in (buildings or [])
                if building.get("roof_surface_source") == "metric-dsm-sampled"
            ),
            "roof_approximate": sum(
                1 for building in (buildings or [])
                if building.get("geometry_quality") == "approximate"
            ),
            "source_gsd_m": None if source_gsd_m is None else float(source_gsd_m),
        },
    })
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if progress_callback:
        progress_callback({"fraction": 1.0, "sub": "GLB and metadata written"})
    return {"output_path": str(output_path), "metadata_path": str(metadata_path), "metadata": metadata}


def main():

    parser = argparse.ArgumentParser(
        description="ASTERRA DSM → textured GLB reconstruction"
    )

    parser.add_argument(
        "--dsm",
        required=True,
        help="ASTERRA metric DSM GeoTIFF"
    )

    parser.add_argument(
        "--rgb",
        required=True,
        help="RGB GeoTIFF"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output GLB path"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Maximum visualization raster dimension"
    )

    args = parser.parse_args()

    dsm_path = Path(args.dsm)
    rgb_path = Path(args.rgb)
    output_path = Path(args.output)

    result = run_reconstruction(dsm_path, rgb_path, output_path, size=args.size)
    print(f"GLB written: {result['output_path']}")
    print(f"Metadata written: {result['metadata_path']}")


if __name__ == "__main__":
    main()
