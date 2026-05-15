"""
§3.2 / §9.3: 프로젝트 전역 설정.
"""

# JSON keys
X_KEY = "two_theta_values"
Y_KEY = "intensities"

# Project root
PROJECT_ROOT = r"c:\xrd_digitizer_v1"

# Data paths
DATA_ROOT = rf"{PROJECT_ROOT}\data"
SOURCE_JSON_ROOT = rf"{DATA_ROOT}\source_json"
METADATA_DIR = rf"{DATA_ROOT}\metadata"
RENDERED_CLEAN_DIR = rf"{DATA_ROOT}\rendered_clean"
RENDERED_STYLED_DIR = rf"{DATA_ROOT}\rendered_styled"
RENDERED_REAL_LIKE_DIR = rf"{DATA_ROOT}\rendered_real_like"
GT_DIR = rf"{DATA_ROOT}\gt"
MANIFESTS_DIR = rf"{DATA_ROOT}\manifests"

# §9 Engine output paths
OUTPUTS_DIR = rf"{PROJECT_ROOT}\outputs"
EXAMPLES_DIR = rf"{PROJECT_ROOT}\examples"

# §5.2 Canvas constants
CANVAS_W = 1200
CANVAS_H = 900
PLOT_BOX_DEFAULT = [170, 90, 1120, 780]

# §9.4 Debug output file names
DEBUG_FILE_NAMES = [
    "overlay", "color_mask", "combined_mask", "skeleton",
    "candidate_map", "trace_path", "peaks_overlay", "debug.json",
]
