"""Configuration constants and runtime Config dataclass for SPHERE-3 pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import argparse
import os

# ============================================================================
# CONSTANTS (from main.py lines 60-106)
# ============================================================================

# Geometric constants
DETECTOR_FOCAL_LENGTH = 330.0
PIXEL_SKIP = 7
GEOMETRY_TYPE = 'pix'

# Physical thresholds
MAX_CENTER_DISTANCE = 400.0
MAX_PEAK_DISTANCE = 330.0
MIN_TOTAL_INTENSITY_DEFAULT = 10.0

# Integration parameters
INTEGRATION_LOWER_BOUND = 0.0
INTEGRATION_UPPER_BOUND = 330.0
INTEGRATION_ABS_ERROR = 1.49e-08
INTEGRATION_REL_ERROR = 1.49e-08

# Optimization parameters
MINUIT_STRATEGY_FAST = 1
MINUIT_STRATEGY_PRECISE = 2
MINUIT_CHI2_THRESHOLD = 2.0

# Minuit parameter limits
#LIMIT_P0_FACTOR = 2.0
#LIMIT_P1_MIN = -1.0
#LIMIT_P1_MAX = 1.0
#LIMIT_P2_MIN = -0.1
#LIMIT_P2_MAX = 0.1
#LIMIT_P3_MIN = -1e-3
#LIMIT_P3_MAX = 1e-3
#LIMIT_P4_MIN = 0.0
#LIMIT_P4_MAX = 1e6
#LIMIT_P5_MIN = -1.0
#LIMIT_P5_MAX = 3.0
#LIMIT_P6_MIN = -0.1
#LIMIT_P6_MAX = 0.1
#LIMIT_RCH_MIN = 10.0
#LIMIT_RCH_MAX = 400.0
#LIMIT_SW_MIN = 1.0
#LIMIT_SW_MAX = 100.0
#LIMIT_X0_MIN = -300.0
#LIMIT_X0_MAX = 300.0
#LIMIT_Y0_MIN = -300.0
#LIMIT_Y0_MAX = 300.0
LIMIT_P0_FACTOR = 1.5
LIMIT_P1_MIN = 0
LIMIT_P1_MAX = 10.0
LIMIT_P2_MIN = 0
LIMIT_P2_MAX = 0.2
LIMIT_P3_MIN = 0
LIMIT_P3_MAX = 0
LIMIT_P4_MIN = 0.0
LIMIT_P4_MAX = 0
LIMIT_P5_MIN = 0
LIMIT_P5_MAX = 0
LIMIT_P6_MIN = 0
LIMIT_P6_MAX = 0
LIMIT_RCH_MIN = 10.0
LIMIT_RCH_MAX = 400.0
LIMIT_SW_MIN = 1.0
LIMIT_SW_MAX = 100.0
LIMIT_X0_MIN = -330.0
LIMIT_X0_MAX = 330.0
LIMIT_Y0_MIN = -330.0
LIMIT_Y0_MAX = 330.0

# Data processing
N_TOP_PEAKS = 10
MESH_RANGE = 325
MESH_STEP = 1

# Background
DEFAULT_BG_SAMPLE = 100

RESULT_COLUMNS = [
    'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'R_ch', 'sw', 'x0', 'y0',
    'fval', 'chi2_ndf', 'max_abs_d', 'mean_abs_d',
    'Rc_snow', 'I_max', 'sum',
    'Int', 'err_Int', 'A', 'B','C', 'D', 'COSpl',
]

# Criterion constants
R1_MIN = 50
R1_MAX = 110
R1_STEP = 2
R2_MIN = 110
R2_MAX = 270
R2_STEP = 2
RADIAL_STEP = 1.0
SEP_BORDER_MIN = -0.2
SEP_BORDER_MAX = 1.5
SEP_BORDER_STEPS = 1000
DEFAULT_ANOMALY_THRESHOLD = 10
# введу границы критерия заранее, возможно нужно читать из файла, кст. Но это позже)))
Rcri1=
Rcri2=

# Default paths
DEFAULT_PIXEL_DATA_PATH = Path('SPHERE3_pixel_data_A.dat')
DEFAULT_MOSHITS_BASE_DIR = Path('/Users/vladimirivanov/Projects/SPHERE/moshits_base/small_sample_with')
DEFAULT_MOSHITS_BG_BASE_DIR = Path('/Users/vladimirivanov/Projects/SPHERE/moshits_base/bg_moshits')


# ============================================================================
# CONFIG DATACLASS
# ============================================================================

@dataclass
class Config:
    moshits_root: Path
    bg_root: Path
    pixel_path: Path
    workers: int
    files_limit: Optional[int]
    min_intensity: float
    skip_vis: bool
    bg_sample: int
    smoke: bool
    profile: bool
    data_root: Optional[Path]
    output: str
    exclude_energies: frozenset[str] = frozenset()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description='SPHERE-3 processing pipeline')
    parser.add_argument('--moshits-root', type=Path, default=DEFAULT_MOSHITS_BASE_DIR)
    parser.add_argument('--bg-root', type=Path, default=DEFAULT_MOSHITS_BG_BASE_DIR)
    parser.add_argument('--pixel-path', type=Path, default=DEFAULT_PIXEL_DATA_PATH)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument('--files-limit', type=int, default=None)
    parser.add_argument('--min-intensity', type=float, default=MIN_TOTAL_INTENSITY_DEFAULT)
    parser.add_argument('--skip-vis', action='store_true')
    parser.add_argument('--bg-sample', type=int, default=DEFAULT_BG_SAMPLE)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--data-root', type=Path, default=None,
                        help='Root of hierarchical binary data ({particle}/{energy}/{angle}/{height}/events.bin)')
    parser.add_argument('--output', '-o', default='results.parquet',
                        help='Output parquet file path (default: results.parquet)')
    parser.add_argument('--exclude-energy', nargs='*', default=[],
                        help='Energy levels to exclude (e.g. --exclude-energy 1PeV)')
    args = parser.parse_args()
    return Config(
        moshits_root=args.moshits_root,
        bg_root=args.bg_root,
        pixel_path=args.pixel_path,
        workers=args.workers,
        files_limit=args.files_limit,
        min_intensity=args.min_intensity,
        skip_vis=args.skip_vis,
        bg_sample=args.bg_sample,
        smoke=args.smoke,
        profile=args.profile,
        data_root=args.data_root,
        output=args.output,
        exclude_energies=frozenset(args.exclude_energy),
    )
