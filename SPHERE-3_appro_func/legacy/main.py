import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from iminuit import Minuit
from scipy.integrate import quad
from scipy.integrate import IntegrationWarning
import warnings
from functools import partial
import math
import random
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Optional, Iterable, Tuple, Dict, Any

# Numba is often lagging behind new CPython releases; make it optional for 3.14+
_NUMBA_IMPORT_ERR = None
try:
    if os.environ.get('NUMBA_DISABLED'):
        raise ImportError('disabled via NUMBA_DISABLED env')
    import numba  # type: ignore
    NUMBA_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - import guard
    NUMBA_AVAILABLE = False
    _NUMBA_IMPORT_ERR = _exc

    class _NoNumba:
        def njit(self, *args, **kwargs):
            # If used as @njit without params, return the function unchanged
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def deco(fn):
                return fn

            return deco

    numba = _NoNumba()  # type: ignore

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Warn once if numba is unavailable (common on fresh Python 3.14 installs)
if not NUMBA_AVAILABLE and _NUMBA_IMPORT_ERR is not None:
    logger.warning(
        "Numba not available (%s); falling back to pure NumPy/Python paths. Performance may be lower.",
        _NUMBA_IMPORT_ERR,
    )

# Geometric constants
DETECTOR_FOCAL_LENGTH = 330.0  # focal length for coordinate transformation
PIXEL_SKIP = 7  # number of pixels to skip for segment coordinates
GEOMETRY_TYPE = 'pix'  # geometry type for processing ('pix', 'seg', 'seg_center')

# Physical thresholds
MAX_CENTER_DISTANCE = 400.0  # maximum distance from center
MAX_PEAK_DISTANCE = 270.0  # maximum distance for peak from center
MIN_TOTAL_INTENSITY = 10.0  # minimum total intensity threshold (can be overridden via CLI)

# Integration parameters
INTEGRATION_LOWER_BOUND = 0.0
INTEGRATION_UPPER_BOUND = 330.0
INTEGRATION_ABS_ERROR = 1.49e-08
INTEGRATION_REL_ERROR = 1.49e-08

# Optimization parameters
MINUIT_STRATEGY = 2
INITIAL_PARAM_P4 = 1e-2
INITIAL_PARAM_S = 1.0

# Minuit parameter limits (6-parameter model, background pre-subtracted)
LIMIT_P0_FACTOR = 2.0
LIMIT_P1_MIN = 0.0
LIMIT_P1_MAX = 1e-2
LIMIT_P4_MIN = 0.0
LIMIT_P4_MAX = 1e1
LIMIT_S_MIN = 0.5
LIMIT_S_MAX = 1.5
LIMIT_X0_MIN = -300.0
LIMIT_X0_MAX = 300.0
LIMIT_Y0_MIN = -300.0
LIMIT_Y0_MAX = 300.0

# Data processing
N_TOP_PEAKS = 10  # number of top peaks for center calculation
MESH_RANGE = 325  # range for mesh grid
MESH_STEP = 1  # step for mesh grid

# File paths (can be overridden by CLI arguments)
DEFAULT_PIXEL_DATA_PATH = Path('SPHERE3_pixel_data_A.dat')
DEFAULT_MOSHITS_BASE_DIR = Path('/Users/vladimirivanov/Projects/SPHERE/moshits_base/small_sample_with')
DEFAULT_MOSHITS_BG_BASE_DIR = Path('/Users/vladimirivanov/Projects/SPHERE/moshits_base/bg_moshits')

# Background sampling
DEFAULT_BG_SAMPLE = 100  # number of bg files to sample when estimating level

# ============================================================================
# DETECTOR COORDINATES LOADING
# ============================================================================

def _list_data_files(directory, prefix=None):
    """
    Return list of data files without extensions and skip hidden service files.
    """
    if not os.path.exists(directory):
        logger.warning("Directory does not exist: %s", directory)
        return []

    files = []
    for f in os.listdir(directory):
        if f.startswith('.'):
            continue  # skip .DS_Store and other hidden files
        if Path(f).suffix != '':
            continue  # skip files with extensions
        if prefix and not f.startswith(prefix):
            continue
        files.append(f)
    return files

def load_detector_coordinates(pixel_data_path=None):
    """
    Load detector pixel and segment coordinates from file.

    This function loads the detector geometry and transforms coordinates
    to the proper reference frame. Used by both worker processes and
    any code that needs detector geometry

    Args:
        pixel_data_path: Path to pixel coordinate data file.
                        If None, uses DEFAULT_PIXEL_DATA_PATH.

    Returns:
        Tuple (coord_pix, coord_seg):
            - coord_pix: DataFrame with pixel coordinates (x, y)
            - coord_seg: DataFrame with segment coordinates (x, y)

    Note:
        Coordinates are transformed using DETECTOR_FOCAL_LENGTH and
        segments are sampled every PIXEL_SKIP pixels.
    """
    if pixel_data_path is None:
        pixel_data_path = DEFAULT_PIXEL_DATA_PATH

    # Load raw coordinates
    coord_pix = pd.read_csv(
        pixel_data_path, header=None, sep=r'\s+',
        names=['x', 'y', 'z', 'vx', 'xy', 'vz', 'phi', 'theta']
    ).drop(columns=['vx', 'xy', 'vz', 'phi', 'theta'])

    # Transform to detector reference frame
    coord_pix['x'] = coord_pix['x'] / coord_pix['z'] * DETECTOR_FOCAL_LENGTH
    coord_pix['y'] = coord_pix['y'] / coord_pix['z'] * DETECTOR_FOCAL_LENGTH
    coord_pix = coord_pix.drop(columns=['z'])

    # Create segment coordinates by sampling pixels
    coord_seg = coord_pix.iloc[::PIXEL_SKIP].reset_index(drop=True)

    return coord_pix, coord_seg


def _create_shared_geometry(pixel_data_path: Path) -> Tuple[Dict[str, Any], Dict[str, shared_memory.SharedMemory]]:
    """Load coordinates once in master and place in shared memory (read-only use).

    Returns metadata for reconstructing arrays in workers and the shm objects to manage lifetime.
    """
    coord_pix, coord_seg = load_detector_coordinates(pixel_data_path)

    shms = {}
    meta = {}

    for name, df in (('pix', coord_pix), ('seg', coord_seg)):
        for axis in ('x', 'y'):
            arr = df[axis].to_numpy()
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr[:]  # copy into shm
            shms[f'{name}_{axis}'] = shm
            meta[f'{name}_{axis}'] = {
                'name': shm.name,
                'shape': arr.shape,
                'dtype': str(arr.dtype),
            }
    return meta, shms


def _attach_shared_geometry(meta: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct coord DataFrames from shared memory blocks inside worker."""
    arrays = {}
    shms_local = {}
    for key, info in meta.items():
        shm = shared_memory.SharedMemory(name=info['name'])
        shms_local[key] = shm
        arrays[key] = np.ndarray(info['shape'], dtype=np.dtype(info['dtype']), buffer=shm.buf)

    coord_pix = pd.DataFrame({'x': arrays['pix_x'], 'y': arrays['pix_y']})
    coord_seg = pd.DataFrame({'x': arrays['seg_x'], 'y': arrays['seg_y']})

    # Keep shm objects referenced in module globals to avoid premature GC/close
    _worker_shms = getattr(_attach_shared_geometry, '_worker_shms', [])
    _worker_shms.extend(list(shms_local.values()))
    _attach_shared_geometry._worker_shms = _worker_shms
    return coord_pix, coord_seg


def _release_shared_geometry(shms: Dict[str, shared_memory.SharedMemory]):
    for shm in shms.values():
        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            continue


# ============================================================================
# INSTANCE POOL EXECUTOR - Optimized parallel processing
# ============================================================================

# Global worker state (initialized once per worker process)
_worker_coord_pix: Optional[pd.DataFrame] = None
_worker_coord_seg: Optional[pd.DataFrame] = None


def _load_background_events(bg_dir):
    """
    Load all background events from directory into memory.

    Args:
        bg_dir: Path to directory containing background event files

    Returns:
        List of tuples (data_pix, data_seg) for each background event
    """
    if not os.path.exists(bg_dir):
        logger.warning(f"Background directory does not exist: {bg_dir}")
        return []

    try:
        # Get list of all files in background directory
        bg_files = _list_data_files(bg_dir)

        if not bg_files:
            logger.warning(f"No background files found in {bg_dir}")
            return []

        logger.info(f"Loading {len(bg_files)} background events into memory...")
        bg_data = []

        for bg_file in bg_files:
            try:
                bg_file_path = bg_dir / bg_file
                data_pix, data_seg = load_data(bg_file_path)
                bg_data.append((data_pix, data_seg))
            except Exception as e:
                logger.warning(f"Failed to load background file {bg_file}: {e}")
                continue

        logger.info(f"Successfully loaded {len(bg_data)} background events")
        return bg_data

    except Exception as e:
        logger.error(f"Error loading background events from {bg_dir}: {e}")
        return []


def _merge_signal_and_background(signal_data_pix, signal_data_seg, bg_data_pix, bg_data_seg):
    """
    Merge signal and background event data by concatenating pixel lists.

    Args:
        signal_data_pix: Signal event pixel data
        signal_data_seg: Signal event segment data
        bg_data_pix: Background event pixel data
        bg_data_seg: Background event segment data

    Returns:
        Tuple (merged_data_pix, merged_data_seg) with combined events
    """
    # Concatenate signal and background data
    merged_data_pix = pd.concat([signal_data_pix, bg_data_pix], ignore_index=True)
    merged_data_seg = pd.concat([signal_data_seg, bg_data_seg], ignore_index=True)

    logger.debug(f"Merged signal ({len(signal_data_pix)} pixels) + background ({len(bg_data_pix)} pixels) = {len(merged_data_pix)} pixels")

    return merged_data_pix, merged_data_seg


def _init_worker_process(pixel_data_path, bg_level, min_intensity, shared_meta=None):
    """
    Initialize worker process with detector coordinates.
    Called once when worker process starts.

    Args:
        pixel_data_path: Path to pixel coordinate data file
        bg_level: Precomputed background level to subtract
    """
    global _worker_coord_pix, _worker_coord_seg, MIN_TOTAL_INTENSITY

    logger.info(f"Initializing worker process {mp.current_process().pid}")

    if shared_meta:
        # Attach to shared memory blocks created by master
        _worker_coord_pix, _worker_coord_seg = _attach_shared_geometry(shared_meta)
    else:
        # Load coordinates once per worker using shared function
        _worker_coord_pix, _worker_coord_seg = load_detector_coordinates(pixel_data_path)
    process_file_wrapper_optimized.bg_level = bg_level
    # allow per-run override of intensity cut
    MIN_TOTAL_INTENSITY = min_intensity

    logger.info(
        "Worker %s initialized with %d pixels (bg_level=%.6f)",
        mp.current_process().pid,
        len(_worker_coord_pix),
        bg_level,
    )


def _worker_process_file(file_path, first_idx, second_idx, bg_level):
    """
    Process a single file in worker process using pre-loaded coordinates.

    Args:
        file_path: Path to event data file
        first_idx: First index for step calculation
        second_idx: Second index for step calculation

    Returns:
        Tuple with processing results or None
    """
    global _worker_coord_pix, _worker_coord_seg

    idx = first_idx * 100 + second_idx

    try:
        params_in = get_first_row(file_path)
        center_distance = np.sqrt(params_in['x_center'] ** 2 +
                                 params_in['y_center'] ** 2)
        if center_distance > MAX_CENTER_DISTANCE:
            logger.debug(f"Event {idx} filtered: center distance {center_distance:.1f} > {MAX_CENTER_DISTANCE}")
            return None

        data_pix, data_seg = load_data(file_path)

        # Use worker's preloaded coordinates instead of global ones
        ldf, _, _, x_peak, y_peak = _get_integral_worker(data_pix, which=GEOMETRY_TYPE, bg_level=bg_level)

        if ldf['I'].sum() < MIN_TOTAL_INTENSITY:
            logger.debug(f"Event {idx} filtered: total intensity {ldf['I'].sum():.1f} < {MIN_TOTAL_INTENSITY}")
            return None

        peak_distance = np.sqrt(x_peak**2 + y_peak**2)
        if peak_distance > MAX_PEAK_DISTANCE:
            logger.debug(f"Event {idx} filtered: peak distance {peak_distance:.1f} > {MAX_PEAK_DISTANCE}")
            return None

        minuit_vals = minimize(ldf, x_peak, y_peak)
        obs = ldf['I'].to_numpy(dtype=float)
        pred = func(
            minuit_vals['p0'], minuit_vals['p1'], minuit_vals['p4'],
            minuit_vals['s'], minuit_vals['x0'], minuit_vals['y0'],
            ldf['x'].values, ldf['y'].values
        )

        nonzero = obs > 0
        obs_nz = obs[nonzero]
        pred_nz = pred[nonzero]

        rmse = np.sqrt(np.mean((pred_nz - obs_nz)**2))
        ss_tot = np.sum((obs_nz - obs_nz.mean())**2)
        ss_res = np.sum((pred_nz - obs_nz)**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        # Integration for LDF (background already subtracted before fitting)
        p0, p1, p4, s, _, _ = unpack_minuit_vals(minuit_vals)
        integral = integrate_function(p0, p1, p4, s)
        int_val, int_err = (integral if integral is not None else (float('nan'), float('nan')))

        # Diagnostic logging for quality metrics
        logger.debug(
            f"Event {idx}: RMSE={rmse:.2f}, R²={r2:.3f}, "
            f"I_total={ldf['I'].sum():.1f}, I_max={ldf['I'].max():.1f}, "
            f"valid={minuit_vals.get('valid', False)}"
        )

        # Warn if fit quality is poor
        if np.isnan(r2) or r2 < 0.5:
            logger.warning(f"Event {idx}: Poor fit quality (R²={r2:.3f})")

        return (
            idx,
            file_path,
            params_in,
            minuit_vals,
            float(rmse),
            float(r2),
            float(obs.max()),
            float(obs.sum()),
            float(int_val),
            float(int_err),
        )

    except Exception as e:
        logger.error(f"Error processing event {idx} from {file_path}: {e}")
        return None


def _get_integral_worker(data: pd.DataFrame, which='seg_center', bg_level=0.0):
    """
    Worker-specific version of get_integral that uses preloaded coordinates.

    Args:
        data: DataFrame with detector hits
        which: Type of processing ('pix', 'seg', or 'seg_center')
        bg_level: Background level to subtract (hits per pixel/segment)

    Returns:
        Tuple (ldf, x_center, y_center, x_peak_mean, y_peak_mean) or None
    """
    global _worker_coord_pix, _worker_coord_seg

    def _build_ldf_from_coord(coord, data_):
        max_abs_pix = len(coord)
        hits = np.bincount(
            data_['abs_pix'].to_numpy(dtype=np.int64),
            minlength=max_abs_pix
        )
        final = pd.DataFrame({'I': hits})
        final['x'] = coord['x'].to_numpy()
        final['y'] = coord['y'].to_numpy()
        return final

    if data is None:
        logger.warning("No data provided to _get_integral_worker")
        return None

    ldf = pd.DataFrame()
    if which == 'pix':
        ldf = _build_ldf_from_coord(_worker_coord_pix, data)
    elif which == 'seg':
        ldf = _build_ldf_from_coord(_worker_coord_seg, data)
    elif which == 'seg_center':
        max_seg = len(_worker_coord_seg)
        seg_hits = np.bincount(
            data['seg'].to_numpy(dtype=np.int64), minlength=max_seg
        )
        ldf = pd.DataFrame(
            {
                'I': seg_hits,
                'x': _worker_coord_seg['x'].to_numpy(),
                'y': _worker_coord_seg['y'].to_numpy(),
            }
        )

    # Use raw intensities without calibration corrections
    ldf['I'] = ldf['I'].astype(float)

    # Subtract statistical background from each element
    if bg_level > 0:
        I_before = ldf['I'].copy()
        ldf['I'] = ldf['I'] - bg_level
        # Ensure no negative values after subtraction (Poisson fluctuations)
        ldf['I'] = ldf['I'].clip(lower=0.0)
        logger.debug(
            f"Background subtraction: bg_level={bg_level:.6f}, "
            f"mean_before={I_before.mean():.2f}, mean_after={ldf['I'].mean():.2f}, "
            f"sum_before={I_before.sum():.1f}, sum_after={ldf['I'].sum():.1f}"
        )

    total_I = ldf['I'].sum()
    if total_I == 0:
        return ldf, 0.0, 0.0, 0.0, 0.0

    x_center = (((ldf['x'] ** 2 * ldf['I']).sum()) / total_I) ** 0.5
    y_center = (((ldf['y'] ** 2 * ldf['I']).sum()) / total_I) ** 0.5
    topN = ldf.nlargest(N_TOP_PEAKS, 'I')
    x_peak_mean = topN['x'].mean()
    y_peak_mean = topN['y'].mean()
    max_I_index = ldf['I'].idxmax()

    if ldf.at[max_I_index, 'x'] < 0:
        x_center = -x_center
    if ldf.at[max_I_index, 'y'] < 0:
        y_center = -y_center
    return ldf, x_center, y_center, x_peak_mean, y_peak_mean


class InstancePoolExecutor:
    """
    Custom executor that initializes worker processes with shared state.
    More efficient than ProcessPoolExecutor for repeated tasks with shared data.
    """

    def __init__(self, max_workers=None, initializer=None, initargs=()):
        """
        Initialize the executor.

        Args:
            max_workers: Maximum number of worker processes
            initializer: Function to initialize each worker
            initargs: Arguments for initializer function
        """
        self.max_workers = max_workers or os.cpu_count()
        self.pool = mp.Pool(
            processes=self.max_workers,
            initializer=initializer,
            initargs=initargs
        )
        logger.info(f"InstancePoolExecutor initialized with {self.max_workers} workers")

    def submit(self, fn, *args, **kwargs):
        """
        Submit a task to the pool.

        Args:
            fn: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            AsyncResult wrapped as Future-like object
        """
        result = self.pool.apply_async(fn, args, kwargs)
        return _PoolFuture(result)

    def shutdown(self, wait=True):
        """
        Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        if wait:
            self.pool.close()
            self.pool.join()
        else:
            self.pool.terminate()
        logger.info("InstancePoolExecutor shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False


class _PoolFuture:
    """Wrapper to make multiprocessing.AsyncResult behave like concurrent.futures.Future"""

    def __init__(self, async_result):
        self._async_result = async_result

    def result(self, timeout=None):
        """Get the result, waiting if necessary."""
        return self._async_result.get(timeout=timeout)

    def done(self):
        """Return True if the task is completed."""
        return self._async_result.ready()

    def exception(self, timeout=None):
        """Get the exception if the task raised one."""
        try:
            self._async_result.get(timeout=0)
            return None
        except Exception as e:
            return e


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def estimate_shape_params(r):
    """
    Estimate initial parameters for the LDF shape function (simplified model: 6 parameters).

    Args:
        r: Array of radial distances

    Returns:
        Tuple of initial parameters (p1, p4, s)
    """
    width = np.percentile(r, 75) - np.percentile(r, 25)
    p1 = 1.0 / (width + 1e-6)
    p4 = INITIAL_PARAM_P4
    s = INITIAL_PARAM_S
    return p1, p4, s


# ============================================================================
# IMPROVED ERROR FUNCTIONS (from optimization_improvements)
# ============================================================================

def error_chi2(I, x, y, sigma, p0, p1, p4, s, x0, y0):
    """
    Chi-square RMSE with Poisson weights (statistically correct).

    For raw pixel counts (background pre-subtracted).
    Uses chi-square approach with Poisson statistics:
    - Each point weighted by 1/sigma_i
    - Sigma estimated as sqrt(I) for Poisson statistics
    - Better for wide range of values

    Args:
        I, x, y: Observed data (background-subtracted counts)
        sigma: Not used (kept for compatibility)
        p0, p1, p4, s: LDF parameters
        x0, y0: Shower center

    Returns:
        Chi-square RMSE value
    """
    F = func(p0, p1, p4, s, x0, y0, x, y)

    chi2 = 0.0
    count = 0

    for i in range(len(I)):
        if I[i] > 0:
            # Poisson error: sigma = sqrt(I)
            sigma_i = math.sqrt(I[i])
            if sigma_i > 0:
                chi2 += ((F[i] - I[i]) / sigma_i) ** 2
                count += 1

    if count > 0:
        return math.sqrt(chi2 / count)
    else:
        return 1e10


def error_chi2_robust(I, x, y, sigma, p0, p1, p4, s, x0, y0):
    """
    Chi-square RMSE with minimum error floor (robust to empty cells).

    For raw pixel counts (background pre-subtracted).
    Adds small offset to prevent division by zero.

    Args:
        I, x, y: Observed data (background-subtracted counts)
        sigma: Not used (kept for compatibility)
        p0, p1, p4, s: LDF parameters
        x0, y0: Shower center

    Returns:
        Chi-square RMSE value (robust)
    """
    F = func(p0, p1, p4, s, x0, y0, x, y)

    chi2 = 0.0
    min_error = 1.0  # Minimum error to prevent over-weighting

    for i in range(len(I)):
        # Add floor to sigma
        sigma_i = math.sqrt(max(I[i], min_error))
        chi2 += ((F[i] - I[i]) / sigma_i) ** 2

    return math.sqrt(chi2 / len(I))


def error_cash(I, x, y, sigma, p0, p1, p4, s, x0, y0):
    """
    Poisson log-likelihood (Cash statistic) — better for sparse counts.
    Returns mean 2*logL to keep scale similar to χ².
    """
    F = func(p0, p1, p4, s, x0, y0, x, y)
    eps = 1e-12
    # avoid log(0) and negative model values
    F = np.clip(F, eps, None)
    term = F - I * np.log(F)
    return 2.0 * np.mean(term)


def error_blend(I, x, y, sigma, p0, p1, p4, s, x0, y0, switch=5.0):
    """
    Blend of Cash (for low counts) and χ² (for high counts).
    """
    F = func(p0, p1, p4, s, x0, y0, x, y)
    eps = 1e-12
    F = np.clip(F, eps, None)
    I = np.asarray(I, dtype=np.float64)
    mask_low = I < switch
    mask_high = ~mask_low

    # Cash part
    cash = F[mask_low] - I[mask_low] * np.log(F[mask_low] + eps)
    # Chi2 part with floor
    chi2 = ((F[mask_high] - I[mask_high]) / np.sqrt(np.maximum(I[mask_high], 1.0))) ** 2

    n = max(len(cash) + len(chi2), 1)
    return 2.0 * (cash.sum() + 0.5 * chi2.sum()) / n


# ============================================================================
# LOW-LEVEL JITTED HELPERS (hot path for Minuit calls)
# ============================================================================


@numba.njit(cache=True)
def _chi2_floor_jit(I, F, min_error):
    """Numba-accelerated chi2 with error floor (no Python loops)."""
    chi2 = 0.0
    n = I.size
    for i in range(n):
        sigma = math.sqrt(I[i]) if I[i] > min_error else math.sqrt(min_error)
        diff = F[i] - I[i]
        chi2 += (diff / sigma) * (diff / sigma)
    return math.sqrt(chi2 / n) if n > 0 else 1e10


@numba.njit(cache=True)
def _blend_stat_jit(I, F, switch, min_error):
    """Cash/chi2 blend; keeps scale compatible with error_blend."""
    eps = 1e-12
    n = I.size
    total = 0.0
    for i in range(n):
        if I[i] < switch:
            f = F[i]
            if f < eps:
                f = eps
            total += f - I[i] * math.log(f)
        else:
            sigma = math.sqrt(I[i]) if I[i] > min_error else math.sqrt(min_error)
            diff = F[i] - I[i]
            total += 0.5 * (diff / sigma) * (diff / sigma)
    return 2.0 * total / n if n > 0 else 1e10


# NumPy fallbacks when numba is unavailable (Python 3.14 fresh installs)
def _chi2_floor_np(I, F, min_error):
    sigma = np.sqrt(np.maximum(I, min_error))
    chi2 = np.mean(((F - I) / sigma) ** 2)
    return float(math.sqrt(chi2)) if chi2 >= 0 and I.size > 0 else 1e10


def _blend_stat_np(I, F, switch, min_error):
    eps = 1e-12
    mask_low = I < switch
    mask_high = ~mask_low
    cash = F[mask_low]
    I_low = I[mask_low]
    cash_term = np.sum(cash - I_low * np.log(np.clip(cash, eps, None)))
    sigma = np.sqrt(np.maximum(I[mask_high], min_error))
    diff = F[mask_high] - I[mask_high]
    chi2_term = np.sum(0.5 * (diff / sigma) ** 2)
    n = I.size
    return float(2.0 * (cash_term + chi2_term) / n) if n > 0 else 1e10


# ============================================================================
# IMPROVED INITIAL PARAMETER ESTIMATION
# ============================================================================

def estimate_shape_params_adaptive(r, I):
    """
    Adaptive estimation of initial LDF parameters from data (background pre-subtracted).

    Estimates parameters based on actual data distribution:
    - p1: From weighted width of distribution
    - p4: From tail fraction
    - s: From tail steepness

    Args:
        r: Radial distances
        I: Intensities (background already subtracted)

    Returns:
        Tuple (p1, p4, s) for 6-parameter model (no p_bg)
    """
    # Filter valid data
    valid_mask = (I > 0) & np.isfinite(r) & np.isfinite(I)
    if not np.any(valid_mask):
        logger.warning("No valid data for adaptive parameter estimation")
        return 1.0 / 50, 0.5, 1.0

    r_valid = r[valid_mask]
    I_valid = I[valid_mask]

    # Weighted statistics
    I_norm = I_valid / I_valid.sum()
    r_mean = np.sum(r_valid * I_norm)
    r_var = np.sum((r_valid - r_mean) ** 2 * I_norm)
    r_std = np.sqrt(r_var)

    # p1: Width of distribution
    width = max(4 * r_std, 10.0)  # Minimum 10mm width
    p1 = 1.0 / width

    # p4, s: Tail behavior
    # Events with I > 10% of max form the "core"
    I_max = I_valid.max()
    core_fraction = np.sum(I_valid > 0.1 * I_max) / len(I_valid)
    tail_fraction = 1.0 - core_fraction

    p4 = max(0.1, 2.0 * tail_fraction)  # More tail → higher p4
    s = 1.0 + 0.5 * tail_fraction  # More tail → higher s

    logger.debug(
        f"Adaptive params: p1={p1:.4f}, "
        f"p4={p4:.3f}, s={s:.3f} "
        f"(r_std={r_std:.1f}, tail={tail_fraction:.1%})"
    )

    return p1, p4, s


def estimate_shape_params_scaled(p1_base, p4_base, s_base, scale):
    """
    Scale initial parameters for restart attempts (background pre-subtracted model).

    Args:
        p1_base, p4_base, s_base: Base LDF parameters
        scale: Scale factor (0.5 to 1.5)

    Returns:
        Tuple (p1, p4, s) - scaled parameters
    """
    p1 = p1_base * scale
    p4 = p4_base * (1.0 + 0.5 * (scale - 1.0))  # Linear blend
    s = s_base + 0.2 * (scale - 1.0)  # Slight variation

    return p1, p4, s


# ============================================================================
# MULTIPLE RESTART OPTIMIZATION
# ============================================================================

def minimize_with_restarts(
    ldf,
    x_center,
    y_center,
    error_func=None,
    n_restarts=3,
    adaptive_init=True,
    verbose=True,
    max_restarts=5,
    rel_tol=1e-3,
    reg_lambda=1e-4,
):
    """
    Minuit minimization with multiple restart strategy.

    Runs optimization several times with different initial conditions,
    returns the best result. Expects background-subtracted data.

    Args:
        ldf: Dictionary with 'I', 'x', 'y' arrays (background already subtracted)
        x_center, y_center: Initial shower center
        error_func: Error function to minimize (default: error_chi2_robust)
        n_restarts: Number of restart attempts (default: 3)
        adaptive_init: Use adaptive parameter estimation (default: True)
        verbose: Print debug info (default: True)

    Returns:
        Dictionary with best optimization results
    """
    if error_func is None:
        error_func = error_chi2_robust

    I_all = np.asarray(ldf['I'], dtype=np.float64)
    x_all = np.asarray(ldf['x'], dtype=np.float64)
    y_all = np.asarray(ldf['y'], dtype=np.float64)

    # drop empty pixels early to shrink problem size for Minuit
    nz_mask = I_all > 0
    I_nz = I_all[nz_mask]
    x_nz = x_all[nz_mask]
    y_nz = y_all[nz_mask]
    if I_nz.size == 0:
        logger.warning("All intensities are zero after bg subtraction; skipping event")
        return None

    best_result = None
    best_fval = float('inf')
    best_attempt = -1

    # Basic stats for dynamic limits and init
    r_all = np.sqrt((x_all - x_center) ** 2 + (y_all - y_center) ** 2)
    Imax = float(I_nz.max()) if I_nz.size else 0.0
    total_I = float(I_nz.sum())
    width_r = np.sqrt(np.average(r_all**2, weights=np.maximum(I_all, 1e-6))) if total_I > 0 else 50.0

    # Robust alternative center guess (weighted mean) + small offsets
    centers = [(x_center, y_center)]
    if total_I > 0:
        cx = np.average(x_nz, weights=np.maximum(I_nz, 1e-6))
        cy = np.average(y_nz, weights=np.maximum(I_nz, 1e-6))
        centers.append((cx, cy))
    offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    best_center = (x_center, y_center)
    best_center_rmse = float('inf')
    # coarse scan to pick starting center
    for base in centers:
        for dx, dy in offsets:
            xc = base[0] + dx
            yc = base[1] + dy
            # crude params for score
            p1_tmp, p4_tmp, s_tmp = estimate_shape_params_adaptive(r_all, I_all)
            p0_tmp = max(np.sqrt(Imax), 1e-6)
            pred = func(p0_tmp, p1_tmp, p4_tmp, s_tmp, xc, yc, x_all, y_all)
            rmse_tmp = np.sqrt(np.mean((pred - I_all) ** 2))
            if rmse_tmp < best_center_rmse:
                best_center_rmse = rmse_tmp
                best_center = (xc, yc)
    x_center, y_center = best_center
    r_all = np.sqrt((x_all - x_center) ** 2 + (y_all - y_center) ** 2)
    width_r = np.sqrt(np.average(r_all**2, weights=np.maximum(I_all, 1e-6))) if total_I > 0 else width_r

    max_attempts = max(max_restarts, n_restarts)
    for attempt in range(max_attempts):
        try:
            # Estimate initial parameters
            if adaptive_init and attempt == 0:
                p1, p4, s = estimate_shape_params_adaptive(r_all, I_all)
            elif adaptive_init:
                scale = 0.7 + attempt * 0.2  # 0.7, 0.9, 1.1, ...
                p1_base, p4_base, s_base = estimate_shape_params_adaptive(r_all, I_all)
                p1, p4, s = estimate_shape_params_scaled(p1_base, p4_base, s_base, scale)
            else:
                p1, p4, s = estimate_shape_params(r_all)
                if attempt > 0:
                    scale = 0.7 + attempt * 0.2
                    p1 *= scale
                    p4 *= (0.5 + attempt * 0.25)

            if attempt == 0:
                p0_init = np.sqrt(Imax)
            elif attempt == 1:
                p0_init = 0.7 * np.sqrt(Imax)
            elif attempt == 2:
                p0_init = 1.3 * np.sqrt(Imax)
            else:
                p0_init = (0.5 + attempt * 0.2) * np.sqrt(Imax)

            # Choose metric: blend for low-count events, keep jit helpers
            switch_val = 5.0
            use_blend = (error_func == error_chi2_robust and Imax < 20)

            def error_wrap(p0, p1, p4, s, x0, y0):
                r = np.sqrt((x_nz - x0) ** 2 + (y_nz - y0) ** 2)
                F = p0 * p0 / ((1.0 + p1 * r) ** 2) / (1.0 + p4 * np.power(r, s))
                if use_blend:
                    if NUMBA_AVAILABLE:
                        return _blend_stat_jit(I_nz, F, switch_val, 1.0)
                    return _blend_stat_np(I_nz, F, switch_val, 1.0)
                else:
                    if NUMBA_AVAILABLE:
                        return _chi2_floor_jit(I_nz, F, 1.0)
                    return _chi2_floor_np(I_nz, F, 1.0)

            # Create Minuit instance (6 parameters, background pre-subtracted)
            m = Minuit(
                error_wrap,
                p0=p0_init,
                p1=p1,
                p4=p4,
                s=s,
                x0=x_center,
                y0=y_center,
            )

            # Set up limits
            m.strategy = MINUIT_STRATEGY
            p0_cap = LIMIT_P0_FACTOR * max(np.sqrt(Imax), 1e-6)
            # Dynamic upper bound for p1 based on event width
            p1_dynamic_max = min(LIMIT_P1_MAX, max(1e-4, 2.0 / max(width_r, 1.0)))
            m.limits["p0"] = (0, p0_cap)
            m.limits["p1"] = (LIMIT_P1_MIN, p1_dynamic_max)
            m.limits["p4"] = (LIMIT_P4_MIN, LIMIT_P4_MAX)
            m.limits["s"] = (LIMIT_S_MIN, LIMIT_S_MAX)
            m.limits["x0"] = (LIMIT_X0_MIN, LIMIT_X0_MAX)
            m.limits["y0"] = (LIMIT_Y0_MIN, LIMIT_Y0_MAX)

            # Run optimization
            m.simplex()
            m.migrad()
            m.hesse()

            if verbose:
                accuracy_str = f"accuracy={m.accuracy:.4e}" if hasattr(m, 'accuracy') else "accuracy=N/A"
                logger.debug(
                    f"Attempt {attempt+1}/{n_restarts}: fval={m.fval:.4f}, "
                    f"valid={m.valid}, {accuracy_str}"
                )

            # Regularized score for selection
            reg_penalty = reg_lambda * (m.values["p1"] ** 2 + m.values["p4"] ** 2)
            fval_reg = m.fval + reg_penalty

            # Check if this is the best result so far
            if fval_reg < best_fval:
                best_fval = fval_reg
                best_result = m.values.to_dict()
                best_result["fcn"] = m.fval
                best_result["fcn_reg"] = fval_reg
                best_result["valid"] = m.valid
                best_attempt = attempt
            else:
                # early stop if no meaningful improvement
                if attempt >= n_restarts and (abs(fval_reg - best_fval) <= rel_tol * (abs(best_fval) + 1e-8)):
                    break

        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            continue

    if best_result is not None:
        logger.debug(f"Best result from attempt {best_attempt+1} (fval={best_fval:.4f})")
        return best_result
    else:
        logger.error("All minimization attempts failed!")
        return None


def get_fit_quality_metrics(observed, predicted):
    """
    Calculate fit quality metrics.

    Args:
        observed: Array of observed values
        predicted: Array of predicted values

    Returns:
        Dictionary with metrics: rmse, r2, chi2, success
    """
    obs = np.array(observed)
    pred = np.array(predicted)

    # Filter valid points
    valid = (obs > 0) & np.isfinite(pred)
    obs_valid = obs[valid]
    pred_valid = pred[valid]

    if len(obs_valid) == 0:
        return {
            "rmse": float("nan"),
            "r2": float("nan"),
            "chi2": float("nan"),
            "success": False,
        }

    # RMSE
    rmse = np.sqrt(np.mean((pred_valid - obs_valid) ** 2))

    # R²
    ss_tot = np.sum((obs_valid - obs_valid.mean()) ** 2)
    ss_res = np.sum((pred_valid - obs_valid) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')

    # Chi²
    chi2 = np.sum(((pred_valid - obs_valid) / np.sqrt(obs_valid)) ** 2) / len(obs_valid)

    # Success: R² > 0.8
    success = r2 > 0.8

    return {"rmse": rmse, "r2": r2, "chi2": chi2, "success": success}


def calculate_background_level(bg_data_cache, which='pix'):
    """
    Calculate average background level from cached background events.

    For raw pixel counts, returns average hits per pixel/segment across all background events.
    This is the statistical background level to subtract from merged data.

    Args:
        bg_data_cache: List of tuples (data_pix, data_seg) from background events
        which: Type of processing ('pix' or 'seg')

    Returns:
        Float: Average number of background hits per pixel/segment
    """
    global _worker_coord_pix, _worker_coord_seg

    if not bg_data_cache or len(bg_data_cache) == 0:
        logger.warning("No background events available for level calculation")
        return 0.0  # No background if no data

    # Calculate average hits per pixel/segment across all background events
    all_intensities = []

    try:
        for bg_data_pix, bg_data_seg in bg_data_cache:
            if which == 'pix':
                num_pixels = len(_worker_coord_pix) if _worker_coord_pix is not None else 2660
                num_hits = len(bg_data_pix)
            elif which == 'seg':
                num_pixels = len(_worker_coord_seg) if _worker_coord_seg is not None else 379
                num_hits = len(bg_data_seg)
            else:  # 'seg_center'
                num_pixels = 379  # Number of segments
                num_hits = len(bg_data_seg)

            # Average hits per pixel/segment for this event
            avg_hits = num_hits / num_pixels
            all_intensities.append(avg_hits)

        # Use median to be robust against outliers
        bg_level = float(np.median(all_intensities))
        logger.debug(f"Calculated background level: {bg_level:.6f} hits/element from {len(all_intensities)} events")
        return bg_level

    except Exception as e:
        logger.error(f"Error calculating background level: {e}")
        return 0.0


def compute_background_level_from_dir(bg_dir: Path, which: str = 'pix', sample_size: int = DEFAULT_BG_SAMPLE) -> float:
    """
    Estimate background level by sampling background files once on master.

    Args:
        bg_dir: Directory with background event files (no extensions)
        which: 'pix' or 'seg' geometry
        sample_size: number of files to sample for estimation

    Returns:
        Median hits per element across sampled events.
    """
    if not bg_dir.exists():
        logger.warning("Background directory does not exist: %s", bg_dir)
        return 0.0

    files = _list_data_files(bg_dir)
    if not files:
        logger.warning("No background files found in %s", bg_dir)
        return 0.0

    sample = random.sample(files, k=min(sample_size, len(files)))
    levels = []
    for fname in sample:
        try:
            data_pix, data_seg = load_data(bg_dir / fname)
            if which == 'pix':
                levels.append(len(data_pix) / 2660)
            else:
                levels.append(len(data_seg) / 379)
        except Exception as exc:
            logger.debug("Skip bg file %s due to %s", fname, exc)
            continue

    if not levels:
        return 0.0
    level = float(np.median(levels))
    logger.info(
        "Computed bg_level=%.6f from %d/%d files in %s", level, len(levels), len(sample), bg_dir
    )
    return level


@numba.njit
def func(p0, p1, p4, s, x0, y0, x, y):
    """
    LDF model for raw pixel counts (6 parameters, background pre-subtracted).

    Modified to work with raw pixel counts after background subtraction.
    The model represents the shower lateral distribution function.

    Args:
        p0: amplitude (sqrt of intensity at center)
        p1: linear radial parameter (controls core width)
        p4: tail amplitude
        s: tail power (controls tail steepness)
        x0, y0: shower center coordinates
        x, y: pixel/segment coordinates

    Returns:
        model intensity values (raw counts after background subtraction)
    """
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    func_value = p0 ** 2 / ((1 + p1 * r) ** 2) / (1 + p4 * r ** s)
    return func_value


@numba.njit
def error(I, x, y, sigma, p0, p1, p4, s, x0, y0):
    """
    Calculate RMSE between model predictions and observations.

    For raw pixel counts (background pre-subtracted).

    Args:
        I: Observed intensity values (background-subtracted counts)
        x, y: Coordinate arrays
        sigma: Not used (kept for signature compatibility)
        p0, p1, p4, s: Model parameters
        x0, y0: Center coordinates

    Returns:
        RMSE value
    """
    F = func(p0, p1, p4, s, x0, y0, x, y)
    sum2 = 0.0
    num = I.size
    for i in range(num):
        sum2 += (F[i] - I[i])**2
    rmse_lin = math.sqrt(sum2 / num)
    return rmse_lin


class Minuiter:
    """
    Optimizer class for fitting LDF parameters using Minuit.
    Expects background-subtracted data.
    """

    def __init__(self, ldf, x_center, y_center):
        """
        Initialize the optimizer.

        Args:
            ldf: Dictionary with 'I', 'x', 'y' arrays (background already subtracted)
            x_center: Initial x-coordinate of shower center
            y_center: Initial y-coordinate of shower center
        """
        self.ldf = {key: np.array(value) for key, value in ldf.items()}
        self.x_center = x_center
        self.y_center = y_center

    def minimize(self):
        """
        Perform minimization to find optimal LDF parameters using improved method.

        Returns:
            Dictionary with optimized parameters and function value
        """
        result = minimize_with_restarts(
            self.ldf,
            self.x_center,
            self.y_center,
            error_func=error_chi2_robust,
            n_restarts=3,  # Increased from 1 to 3 for better convergence
            adaptive_init=True,
        )
        return result


@numba.njit
def func_int(r, p0, p1, p4, s):
    """Integrand function for radial LDF integration."""
    return p0 ** 2 * r / (((1 + p1 * r) ** 2) * (1 + p4 * r ** s))


def integrate_function(p0, p1, p4, s, a=None, b=None):
    """
    Integrate the LDF function over radial distance.

    Args:
        p0-p4, s: Model parameters
        a: Lower integration bound (default from config)
        b: Upper integration bound (default from config)

    Returns:
        Tuple (integral_value, error) or None if integration fails
    """
    if a is None:
        a = INTEGRATION_LOWER_BOUND
    if b is None:
        b = INTEGRATION_UPPER_BOUND

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            integral = quad(
                lambda r: func_int(r, p0, p1, p4, s),
                a, b,
                epsabs=INTEGRATION_ABS_ERROR,
                epsrel=INTEGRATION_REL_ERROR
            )
            return integral
        except IntegrationWarning:
            return None


def count_files_without_extension(directory_path, prefix, files_num=None):
    """
    Count and list files without extension in a directory.

    Args:
        directory_path: Path to directory
        prefix: File prefix to filter
        files_num: Optional limit on number of files to return

    Returns:
        Tuple (total_count, file_list)
    """
    if not os.path.exists(directory_path):
        logger.error(f"Directory does not exist: {directory_path}")
        return 0, []

    try:
        files_ = _list_data_files(directory_path, prefix=prefix)
        if files_num is not None and files_num > 0:
            files_ = random.sample(files_, min(files_num, len(files_)))
        logger.info(f"Found {len(files_)} files with prefix '{prefix}' in {directory_path}")
        return len(files_), files_
    except OSError as e:
        logger.error(f"Error reading directory {directory_path}: {e}")
        return 0, []

def load_data(file):
    """
    Load detector data from file.

    Args:
        file: Path to data file

    Returns:
        Tuple (data_pix, data_seg) - pixel and segment data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data format is invalid
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"Data file not found: {file}")

    try:
        data_pix = pd.read_csv(file, header=None, sep=r'\s+', skiprows=1,
                               names=['seg', 'pix', 'x', 'y', 'z', 't', 'vx', 'xy', 'vz',
                                      'origin', 'ii', 'jj', 'mmm', 'xx', 'yy', 'tt'])
        data_seg = data_pix.copy().rename(columns={'seg': 'abs_pix'})
        data_pix['abs_pix'] = data_pix['seg'] * PIXEL_SKIP + data_pix['pix']

        logger.debug(f"Loaded {len(data_pix)} pixel events from {file}")
        return data_pix, data_seg
    except Exception as e:
        logger.error(f"Error loading data from {file}: {e}")
        raise ValueError(f"Invalid data format in {file}: {e}")


def get_first_row(file):
    """
    Parse metadata from the first row of data file.

    Args:
        file: Path to data file

    Returns:
        Dictionary with metadata (clone_num, h, x_center, y_center, event_num)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If first row format is invalid
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"Data file not found: {file}")

    try:
        with open(file, 'r') as f:
            first_row = f.readline().rstrip()
        first_row_values = first_row.split()

        if len(first_row_values) < 5:
            raise ValueError(f"Expected at least 5 values, got {len(first_row_values)}")

        return {
            'clone_num': int(first_row_values[0]),
            'h': float(first_row_values[1]),
            'x_center': float(first_row_values[2]),
            'y_center': float(first_row_values[3]),
            'event_num': int(first_row_values[4])
        }
    except (IndexError, ValueError) as e:
        logger.error(f"Error parsing first row from {file}: {e}")
        raise ValueError(f"Error parsing the first row: {e}")


def unpack_minuit_vals(minuit_vals):
    """
    Unpack Minuit optimization results.

    Args:
        minuit_vals: Dictionary with optimization results

    Returns:
        Tuple of parameters (p0, p1, p4, s, x0, y0)
    """
    return (minuit_vals.get('p0'), minuit_vals.get('p1'), minuit_vals.get('p4'), minuit_vals['s'],
            minuit_vals['x0'], minuit_vals['y0'])


def minimize(ldf, x_center, y_center):
    """
    Convenience wrapper for Minuiter minimization.

    Args:
        ldf: LDF data dictionary (background already subtracted)
        x_center, y_center: Initial shower center coordinates

    Returns:
        Dictionary with optimized parameters
    """
    temp = Minuiter(ldf, x_center, y_center)
    return temp.minimize()


def process_file_wrapper_optimized(moshits_dir, file):
    """
    Optimized wrapper function for parallel file processing with InstancePoolExecutor.
    Only passes minimal data (directory path and filename) instead of entire Event object.

    Args:
        moshits_dir: Path to moshits directory
        file: Filename to process

    Returns:
        Processing result tuple or None
    """
    first_idx = int(str(file).split('_')[-3])
    second_idx = int(str(file).split('c')[-1])
    current_file = moshits_dir / Path(file)
    return _worker_process_file(current_file, first_idx, second_idx, process_file_wrapper_optimized.bg_level)

process_file_wrapper_optimized.bg_level = 0.0  # set by master before pool start

class Event:
    """
    Main class for processing cosmic ray shower events.

    Handles loading, processing, and analyzing detector data files
    for a collection of shower events.
    """

    def __init__(self, folder_name: str, moshits_root: Path, pixel_data_path: Path,
                 bg_level: float, files_limit: Optional[int], workers: int,
                 chunk_size: Optional[int], shared_meta: Optional[Dict[str, Any]]):
        """
        Initialize Event processor.

        Args:
            folder_name: Name of folder containing event data files
            moshits_root: Root directory with moshits_* folders
            pixel_data_path: Path to detector geometry file
            bg_level: Precomputed background level
            files_limit: Limit number of files to process (None = all)
            workers: Number of worker processes
        """
        self.folder_name = folder_name
        self.moshits_dir = moshits_root / folder_name
        self.data_pix = None
        self.data_seg = None
        self.current_file = None
        self.params_in = {
            'clone_num': 0,
            'h': 0,
            'x_center': 0.,
            'y_center': 0.,
            'event_num': 0
        }
        self.x_center = 0
        self.y_center = 0
        self.i = None
        self.x = None
        self.y = None
        self.ldf = pd.DataFrame()
        self.minuit_vals = None
        # Use list to accumulate results instead of pre-allocated DataFrame
        self.results_list = []
        self.file_list = None
        self.files_num = 0
        x = np.arange(-MESH_RANGE, MESH_RANGE, MESH_STEP)
        y = np.arange(-MESH_RANGE, MESH_RANGE, MESH_STEP)
        self.x_mesh, self.y_mesh = np.meshgrid(x, y)
        logger.info(
            "Initialized Event processor for folder=%s workers=%s limit=%s pixel_path=%s bg_level=%.6f",
            folder_name, workers, files_limit, pixel_data_path, bg_level
        )
        self.bg_level = bg_level
        self.files_limit = files_limit
        self.workers = workers
        self.pixel_data_path = pixel_data_path
        self.chunk_size = chunk_size
        self.shared_meta = shared_meta

    def calculate(self):
        """
        Process all event files in parallel using optimized InstancePoolExecutor.

        Uses InstancePoolExecutor with pre-initialized worker processes for
        efficient parallel processing with minimal data transfer overhead.
        """
        self.files_num, self.file_list = count_files_without_extension(
            self.moshits_dir, 'moshits', files_num=self.files_limit
        )

        if self.files_num == 0:
            logger.warning(f"No files found in {self.moshits_dir}")
            return

        num_workers = self.workers
        logger.info(
            "Processing %d files using InstancePoolExecutor with %d workers (bg_level=%.6f)",
            self.files_num, num_workers, self.bg_level
        )
        logger.info("Initializing worker processes with detector coordinates...")

        with InstancePoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker_process,
            initargs=(self.pixel_data_path, self.bg_level, MIN_TOTAL_INTENSITY, self.shared_meta)
        ) as executor:
            process_file_wrapper_optimized.bg_level = self.bg_level
            worker_fn = partial(process_file_wrapper_optimized, self.moshits_dir)
            # Tune chunksize to reduce IPC overhead (work items are lightweight after masking)
            if self.chunk_size is not None and self.chunk_size > 0:
                chunk = self.chunk_size
            else:
                chunk = max(1, int(len(self.file_list) / (4 * num_workers)))
            for idx, result in enumerate(
                executor.pool.imap_unordered(worker_fn, self.file_list, chunksize=chunk), 1
            ):
                if result is None:
                    continue
                self.save_results_threadsafe(*result)
                if idx % 100 == 0 or idx == self.files_num:
                    logger.info("Processed %d/%d files", idx, self.files_num)

        logger.info(f"Finished processing. Saving results...")
        self.save_params()

    def save_results_threadsafe(self, step, file, params_in, minuit_vals, rmse, r2, I_max, I_sum, Int, err_Int):
        """
        Save processing results for a single event (thread-safe).

        Args:
            step: Event index
            file: Source filename
            params_in: Input parameters from file header
            minuit_vals: Optimized parameters from Minuit
            ldf: LDF data
            rmse: Root mean square error
            r2: R-squared goodness of fit
        """
        result_dict = {
            'step': step,
            'file': str(file).split('/')[-1],
            'rmse': rmse,
            'r2': r2,
            'Rc_snow': np.sqrt(params_in['x_center'] ** 2 + params_in['y_center'] ** 2),
            'I_max': I_max,
            'sum': I_sum,
            'Int': Int,
            'err_Int': err_Int
        }
        result_dict.update(minuit_vals)
        self.results_list.append(result_dict)

    def save_params(self):
        """
        Save all accumulated results to CSV file.
        """
        if not self.results_list:
            logger.warning(f"No results to save for {self.folder_name}")
            return

        param_out = pd.DataFrame(self.results_list)
        output_file = f'{self.folder_name}_params.csv'
        param_out.to_csv(output_file, index=False)
        logger.info(f"Saved {len(param_out)} results to {output_file}")

    def visualize_results(self, debug_dir: str = 'debug_plots'):
        """
        Create visualization plots for optimization results (requires minuit_visualization module).

        Args:
            debug_dir: Directory to save visualization plots
        """

        if not self.results_list:
            logger.warning(f"No results to visualize for {self.folder_name}")
            return

        import os
        os.makedirs(debug_dir, exist_ok=True)

        # Create visualization output subdirectory
        output_subdir = os.path.join(debug_dir, self.folder_name)
        os.makedirs(output_subdir, exist_ok=True)

        try:
            # Convert results to DataFrame
            param_out = pd.DataFrame(self.results_list)

            # Create parameter statistics plots
            logger.info(f"Creating visualization plots for {self.folder_name}...")

            import matplotlib.pyplot as plt

            # Plot 1: Parameter distributions
            param_cols = ['p0', 'p1', 'p4', 's', 'x0', 'y0']
            available_cols = [col for col in param_cols if col in param_out.columns]

            n_plots = len(available_cols)
            n_cols = 3
            n_rows = (n_plots + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows), dpi=100)
            axes = axes.flatten()

            for idx, param in enumerate(available_cols):
                ax = axes[idx]
                values = param_out[param].values

                ax.hist(values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                ax.axvline(values.mean(), color='red', linewidth=2, linestyle='--',
                          label=f'Mean: {values.mean():.4f}')
                ax.set_xlabel(param)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Parameter {param}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for idx in range(n_plots, len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle(f'{self.folder_name} - Parameter Statistics', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plot_path = os.path.join(output_subdir, 'parameter_statistics.png')
            fig.savefig(plot_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved parameter statistics to {plot_path}")
            plt.close(fig)

            # Plot 2: Fit quality metrics
            if 'rmse' in param_out.columns and 'r2' in param_out.columns:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
                axes = axes.flatten()

                # RMSE distribution
                ax = axes[0]
                rmse_values = param_out['rmse'].values
                ax.hist(rmse_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                ax.axvline(rmse_values.mean(), color='red', linewidth=2, linestyle='--',
                          label=f'Mean: {rmse_values.mean():.4f}')
                ax.set_xlabel('RMSE')
                ax.set_ylabel('Frequency')
                ax.set_title('RMSE Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # R² distribution
                ax = axes[1]
                r2_values = param_out['r2'].values
                ax.hist(r2_values, bins=30, edgecolor='black', alpha=0.7, color='seagreen')
                ax.axvline(r2_values.mean(), color='red', linewidth=2, linestyle='--',
                          label=f'Mean: {r2_values.mean():.4f}')
                ax.set_xlabel('R²')
                ax.set_ylabel('Frequency')
                ax.set_title('R² Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # RMSE vs R²
                ax = axes[2]
                ax.scatter(param_out['rmse'].values, param_out['r2'].values, alpha=0.5, s=20)
                ax.set_xlabel('RMSE')
                ax.set_ylabel('R²')
                ax.set_title('RMSE vs R²')
                ax.grid(True, alpha=0.3)

                # Summary statistics
                ax = axes[3]
                ax.axis('off')
                summary_text = (f"STATISTICS SUMMARY\n"
                              f"{'='*40}\n\n"
                              f"Total Events: {len(param_out)}\n\n"
                              f"RMSE\n"
                              f"  Mean: {rmse_values.mean():.6f}\n"
                              f"  Std:  {rmse_values.std():.6f}\n"
                              f"  Min:  {rmse_values.min():.6f}\n"
                              f"  Max:  {rmse_values.max():.6f}\n\n"
                              f"R²\n"
                              f"  Mean: {r2_values.mean():.6f}\n"
                              f"  Std:  {r2_values.std():.6f}\n"
                              f"  Min:  {r2_values.min():.6f}\n"
                              f"  Max:  {r2_values.max():.6f}\n")
                ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

                plt.suptitle(f'{self.folder_name} - Fit Quality Analysis', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plot_path = os.path.join(output_subdir, 'fit_quality.png')
                fig.savefig(plot_path, dpi=100, bbox_inches='tight')
                logger.info(f"Saved fit quality plot to {plot_path}")
                plt.close(fig)

            logger.info(f"Visualization plots created for {self.folder_name}")

        except Exception as e:
            logger.error(f"Error creating visualization plots: {e}", exc_info=True)

def run_smoke_check(pixel_data_path: Path) -> bool:
    """Quick sanity check without full dataset."""
    global _worker_coord_pix, _worker_coord_seg
    _worker_coord_pix, _worker_coord_seg = load_detector_coordinates(pixel_data_path)

    # synthetic tiny event near center
    data = pd.DataFrame({
        'seg': [0, 0, 1, 1, 2],
        'pix': [0, 1, 0, 1, 0],
        'abs_pix': [0, 1, 7, 8, 14]
    })
    ldf, xc, yc, xp, yp = _get_integral_worker(data, which=GEOMETRY_TYPE, bg_level=0.0)
    ok = not ldf.empty and np.isfinite([xc, yc, xp, yp]).all()
    if ok:
        logger.info("Smoke check passed: ldf size=%d xc=%.2f yc=%.2f", len(ldf), xc, yc)
    else:
        logger.error("Smoke check failed")
    return ok


def parse_args():
    parser = argparse.ArgumentParser(description="SPHERE-3 processing pipeline")
    parser.add_argument('--moshits-root', type=Path, default=DEFAULT_MOSHITS_BASE_DIR,
                        help='Root directory containing moshits_* folders')
    parser.add_argument('--bg-root', type=Path, default=DEFAULT_MOSHITS_BG_BASE_DIR,
                        help='Background events directory')
    parser.add_argument('--pixel-path', type=Path, default=DEFAULT_PIXEL_DATA_PATH,
                        help='Detector geometry file')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help='Number of worker processes')
    parser.add_argument('--files-limit', type=int, default=None,
                        help='Limit number of files per class (optional)')
    parser.add_argument('--bg-sample', type=int, default=DEFAULT_BG_SAMPLE,
                        help='How many bg files to sample for bg_level')
    parser.add_argument('--skip-vis', action='store_true', help='Skip visualization step')
    parser.add_argument('--smoke', action='store_true', help='Run quick smoke test and exit')
    parser.add_argument('--min-intensity', type=float, default=MIN_TOTAL_INTENSITY,
                        help='Minimum total intensity threshold (default 700)')
    parser.add_argument('--profile', action='store_true',
                        help='Run cProfile around processing loop for hotspot inspection')
    parser.add_argument('--chunk-size', type=int, default=None,
                        help='Custom chunksize for pool.imap_unordered (default: auto)')
    parser.add_argument('--share-geometry', action='store_true',
                        help='Load detector geometry once in master and share via shared_memory to workers')
    return parser.parse_args()


def validate_inputs(args) -> None:
    if not args.pixel_path.exists():
        raise FileNotFoundError(f"Pixel data file not found: {args.pixel_path}")
    if not args.moshits_root.exists():
        raise FileNotFoundError(f"Moshits root not found: {args.moshits_root}")
    if not args.bg_root.exists():
        logger.warning("Background root %s not found; bg_level will be 0", args.bg_root)


if __name__ == '__main__':
    # Prefer forkserver on macOS/Linux to reduce pickle overhead vs spawn
    try:
        mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass

    args = parse_args()
    validate_inputs(args)

    logger.info("="*60)
    logger.info("Starting cosmic ray shower event processing")
    logger.info("="*60)

    if args.smoke:
        run_smoke_check(args.pixel_path)
        raise SystemExit(0)

    MIN_TOTAL_INTENSITY = args.min_intensity  # runtime override before workers start

    bg_level = compute_background_level_from_dir(args.bg_root, which=GEOMETRY_TYPE, sample_size=args.bg_sample)

    shared_meta = None
    shm_blocks = None
    if args.share_geometry:
        try:
            shared_meta, shm_blocks = _create_shared_geometry(args.pixel_path)
            logger.info("Geometry shared via shared_memory blocks")
        except Exception as exc:
            logger.warning("Failed to create shared geometry (%s); falling back to per-process load", exc)
            shared_meta = None
            shm_blocks = None

    profiler = None
    if args.profile:
        import cProfile, pstats, io
        profiler = cProfile.Profile()
        profiler.enable()

    for folder in ['moshits_p', 'moshits_N', 'moshits_Fe']:
        evt = Event(
            folder,
            args.moshits_root,
            args.pixel_path,
            bg_level,
            args.files_limit,
            args.workers,
            args.chunk_size,
            shared_meta,
        )
        logger.info(f"Processing {evt.folder_name}...")
        evt.calculate()
        if not args.skip_vis:
            logger.info(f"Creating visualizations for {evt.folder_name}...")
            evt.visualize_results()

    if profiler is not None:
        profiler.disable()
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats('cumulative').print_stats(30)
        logger.info("cProfile top30 (cumulative):\n%s", s.getvalue())

    if shm_blocks:
        _release_shared_geometry(shm_blocks)

    logger.info("="*60)
    logger.info("All processing completed successfully")
    logger.info("="*60)
