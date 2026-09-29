import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from rasterio.features import geometry_mask
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

    if roof_surface is not None and not roof_surface.is_empty:
        # The DSM mesh is the measured roof. Temple profiles are appended as a
        # bounded, explicitly approximate architectural prior below.
        measured = roof_surface.copy()
        roof = float(np.max(measured.vertices[:, 2])) + 0.08
        if building.get("geometry_profile") == "temple":
            profile = _temple_roof_mesh(building, ring_xy, roof)
            if profile is not None:
                measured = trimesh.util.concatenate([measured, profile])
        return measured
    ring = [(x, y, roof) for x, y in ring_xy]
    vertices = np.asarray(ring, dtype=np.float32)
    triangles = _triangulate_ring(ring_xy)
    if not triangles:
        return None
    faces = [[a, b, c] for a, b, c in triangles]
    if building.get("geometry_profile") == "temple":
        profile = _temple_roof_mesh(building, ring_xy, roof)
        if profile is not None:
            vertices = np.vstack((vertices, profile.vertices))
            faces.extend((profile.faces + len(ring)).tolist())
    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces):
        return None
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.material = SimpleMaterial(diffuse=(194, 168, 117, 255))
    return mesh


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
    for building in buildings or []:
        roof_surface = None
        roof_stats = {
            "roof_surface_source": "unavailable:no-raw-dsm",
            "roof_valid_fraction": 0.0,
            "geometry_quality": "approximate",
            "profile_source": "bounded-procedural-profile",
        }
        if raw_dsm is not None and raw_transform is not None:
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
        if roof_surface is not None and not roof_surface.is_empty:
            measured_top = float(np.max(roof_surface.vertices[:, 2])) + metadata["vertical_origin_m"]
            if np.isfinite(measured_top):
                building["roof_elevation"] = measured_top
                ground_value = float(building.get("ground_elevation", metadata["vertical_origin_m"]))
                if np.isfinite(ground_value):
                    building["height"] = round(max(0.5, measured_top - ground_value), 3)
                    building["height_source"] = "metric-dsm-sampled"
        mesh = _building_mesh(
            building,
            metadata["origin_x"],
            metadata["origin_y"],
            metadata["vertical_origin_m"],
        )
        if mesh is None:
            continue
        identifier = int(building.get("id", added + 1))
        scene.add_geometry(mesh, node_name=f"BUILDING_{identifier}_{added + 1}", geom_name=f"BUILDING_{identifier}_{added + 1}")
        roof = _building_roof_mesh(
            building,
            metadata["origin_x"],
            metadata["origin_y"],
            metadata["vertical_origin_m"],
            roof_surface=roof_surface,
        )
        if roof is not None:
            if texture_image is not None:
                apply_rgb_texture(roof, texture_image, *(texture_bounds or (None, None)))
            scene.add_geometry(roof, node_name=f"ROOF_BUILDING_{identifier}_{added + 1}", geom_name=f"ROOF_BUILDING_{identifier}_{added + 1}")
        # Add sloped roof overlay when DSM shows significant relief
        sloped_roof = _sloped_roof_mesh(
            building,
            metadata["origin_x"],
            metadata["origin_y"],
            metadata["vertical_origin_m"],
            roof_surface=roof_surface,
        )
        if sloped_roof is not None:
            if texture_image is not None:
                apply_rgb_texture(sloped_roof, texture_image, *(texture_bounds or (None, None)))
            scene.add_geometry(sloped_roof, node_name=f"SLOPED_ROOF_{identifier}_{added + 1}", geom_name=f"SLOPED_ROOF_{identifier}_{added + 1}")
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
        mesh = _road_mesh(feature, metadata["origin_x"], metadata["origin_y"], metadata["vertical_origin_m"])
        if mesh is not None:
            name = f"ROAD_{int(feature.get('id', environment_counts['roads'] + 1))}_{environment_counts['roads'] + 1}"
            scene.add_geometry(mesh, node_name=name, geom_name=name)
            environment_counts["roads"] += 1
    # Vegetation is a draped context layer and must not flatten hills beneath
    # it. Water receives a stable base level; buildings are flattened so their
    # isolated walls do not sit on top of roof-height terrain.
    for layer in ("water",):
        for feature in environment.get(layer) or []:
            mesh = _polygon_mesh(
                feature,
                metadata["origin_x"],
                metadata["origin_y"],
                metadata["vertical_origin_m"],
                colors["water" if layer == "water" else feature.get("class", "park")],
                z_offset=0.12 if layer == "water" else 0.06,
            )
            if mesh is not None:
                name = f"{'WATER' if layer == 'water' else 'VEGETATION'}_{int(feature.get('id', environment_counts[layer] + 1))}_{environment_counts[layer] + 1}"
                scene.add_geometry(mesh, node_name=name, geom_name=name)
                environment_counts[layer] += 1
    for tree in environment.get("trees") or []:
        mesh = _tree_mesh(tree, metadata["origin_x"], metadata["origin_y"], metadata["vertical_origin_m"])
        if mesh is not None:
            name = f"VEGETATION_TREE_{int(tree.get('id', environment_counts['trees'] + 1))}_{environment_counts['trees'] + 1}"
            scene.add_geometry(mesh, node_name=name, geom_name=name)
            environment_counts["trees"] += 1
    for region in (environment.get("semantic_regions") or [])[:300]:
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
    for layer in ("water", "landcover"):
        for feature in environment.get(layer) or []:
            if feature.get("polygon_projected"):
                overrides.append(feature)
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
    terrain_overrides = _terrain_overrides(buildings, environment)
    # Keep this unmodified calibrated grid for measured roof sampling. The
    # terrain call below receives a separate copy with only accepted physical
    # features flattened beneath their isolated scene layers.
    raw_dsm, raw_transform, _raw_crs, _raw_bounds, _raw_nodata, _raw_overrides, _raw_filtered = load_downsampled_dsm(
        dsm_path,
        target_size=size,
        surface_overrides=None,
    )
    mesh, metadata = create_dsm_mesh(
        dsm_path,
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
    metadata.update({
        "source_dsm": str(dsm_path),
        "source_rgb": str(rgb_path),
        "output_glb": str(output_path),
        "visualization_target_size": size,
        "coordinate_system": "Local mesh coordinates",
        "z_units": z_units,
        "horizontal_units": "meters" if z_units == "meters" else "relative grid units",
        "vertical_units": "meters" if z_units == "meters" else "relative units",
        "vertical_origin_m": metadata.get("vertical_origin_m"),
        "texture_valid": True,
        "mesh_valid": True,
        "description": "ASTERRA DSM converted to a textured 3D mesh. Original geospatial reference is preserved in this metadata.",
        "buildings_detected": building_count,
        "building_vertices": building_vertices,
        "building_faces": building_faces,
        "has_buildings": building_count > 0,
        "environment_counts": environment_counts,
        "has_environment": any(
            int(value) > 0
            for key, value in environment_counts.items()
            if key not in {"semantic", "semantic_layers"}
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
