from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
import json


SRTM = Path(
    r"D:\Asterra AI\calibration\dem\srtm\n30_w082_1arc_v3.tif"
)

REFERENCE = Path(
    r"D:\Asterra AI\datasets\Urban3D\train\Inputs\JAX_Tile_004_RGB.tif"
)

OUTPUT = Path(
    r"D:\Asterra AI\calibration\dem\srtm\JAX_Tile_004_SRTM.tif"
)


def main():

    print("=" * 70)
    print("ASTERRA SRTM AOI EXTRACTION")
    print("=" * 70)

    with rasterio.open(REFERENCE) as ref:

        print("Reference CRS:", ref.crs)
        print("Reference size:", ref.width, "x", ref.height)
        print("Reference bounds:", ref.bounds)

        # Reference footprint
        ref_bounds = ref.bounds

        polygon = {
            "type": "Polygon",
            "coordinates": [[
                [ref_bounds.left, ref_bounds.bottom],
                [ref_bounds.right, ref_bounds.bottom],
                [ref_bounds.right, ref_bounds.top],
                [ref_bounds.left, ref_bounds.top],
                [ref_bounds.left, ref_bounds.bottom],
            ]]
        }

        # Convert RGB footprint to SRTM CRS
        with rasterio.open(SRTM) as src:

            polygon_srtm = transform_geom(
                ref.crs,
                src.crs,
                polygon
            )

            clipped, transform = mask(
                src,
                [polygon_srtm],
                crop=True,
                filled=True,
                nodata=src.nodata
            )

            profile = src.profile.copy()

            profile.update({
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": transform,
                "compress": "deflate",
                "predictor": 2
            })

            OUTPUT.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(OUTPUT, "w", **profile) as dst:
                dst.write(clipped)

            data = clipped[0]

            valid = data != src.nodata

            print("\nSRTM AOI:")
            print("Output:", OUTPUT)
            print("Shape:", clipped.shape[1], "x", clipped.shape[2])
            print("CRS:", src.crs)
            print("Resolution:", src.res)
            print("NoData:", src.nodata)
            print("Valid pixels:", int(valid.sum()))

            if valid.any():
                print("Elevation min:", float(data[valid].min()))
                print("Elevation max:", float(data[valid].max()))
                print("Elevation mean:", float(data[valid].mean()))
                print("Elevation std:", float(data[valid].std()))

    print("=" * 70)
    print("SRTM AOI EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()