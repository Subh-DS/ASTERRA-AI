
from pathlib import Path


# ============================================================
# ASTERRA CALIBRATION CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Asterra AI")

CALIBRATION_ROOT = PROJECT_ROOT / "calibration"

DEM_DIR = CALIBRATION_ROOT / "dem"
OUTPUT_DIR = CALIBRATION_ROOT / "outputs"
REPORT_DIR = CALIBRATION_ROOT / "reports"


# Default raster processing settings
DEFAULT_RESAMPLING = "bilinear"

# Output data type
OUTPUT_DTYPE = "float32"


# ASTERRA height product
HEIGHT_PRODUCT_NAME = "ASTERRA nDSM"

# Core reconstruction relationship:
#
#     DSM = DTM + nDSM
#
DSM_FORMULA = "DSM = DTM + nDSM"
