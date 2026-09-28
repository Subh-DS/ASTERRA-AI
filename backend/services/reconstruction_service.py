"""Adapter around the tested DSM-to-mesh and GLB exporter."""

from visualization.generate_3d import run_reconstruction


def reconstruct(
    dsm_path,
    rgb_path,
    output_path,
    metadata_path,
    size=512,
    progress_callback=None,
    z_units="meters",
    buildings=None,
    environment=None,
    source_gsd_m=None,
):
    return run_reconstruction(
        dsm_path=dsm_path,
        rgb_path=rgb_path,
        output_path=output_path,
        metadata_path=metadata_path,
        size=size,
        progress_callback=progress_callback,
        z_units=z_units,
        buildings=buildings,
        environment=environment,
        source_gsd_m=source_gsd_m,
    )
