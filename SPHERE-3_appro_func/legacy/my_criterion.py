import logging
import os

import numpy as np

# Numba is often lagging behind new CPython releases; make it optional
_NUMBA_IMPORT_ERR = None
try:
    if os.environ.get('NUMBA_DISABLED'):
        raise ImportError('disabled via NUMBA_DISABLED env')
    import numba
    NUMBA_AVAILABLE = True
except Exception as _exc:
    NUMBA_AVAILABLE = False
    _NUMBA_IMPORT_ERR = _exc

    class _NoNumba:
        def jit(self, *args, **kwargs):
            # Support @jit(nopython=True) and @jit without params
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def deco(fn):
                return fn

            return deco

    numba = _NoNumba()  # type: ignore
import pandas as pd
import scipy.integrate
from matplotlib import pyplot as plt

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Warn once if numba is unavailable
if not NUMBA_AVAILABLE and _NUMBA_IMPORT_ERR is not None:
    logger.warning(
        "Numba not available (%s); falling back to pure Python. Performance may be lower.",
        _NUMBA_IMPORT_ERR,
    )

# Radial range parameters for optimization
R1_MIN = 50  # minimum inner radius (mm)
R1_MAX = 110  # maximum inner radius (mm)
R1_STEP = 2  # step size for r1 range

R2_MIN = 110  # minimum outer radius (mm)
R2_MAX = 270  # maximum outer radius (mm)
R2_STEP = 2  # step size for r2 range
RADIAL_STEP = 1.0  # step (mm) for precomputed cumulative integral grid

# Integration parameters
INTEGRATION_ABS_ERROR = 1.49e-8  # absolute error tolerance for integration
INTEGRATION_REL_ERROR = 1.49e-8  # relative error tolerance for integration

# Anomaly detection threshold
DEFAULT_ANOMALY_THRESHOLD = 10  # threshold for filtering anomalous criterion values

# Separation optimization parameters
SEP_BORDER_MIN = -0.2  # minimum border value for separation
SEP_BORDER_MAX = 1.5  # maximum border value for separation
SEP_BORDER_STEPS = 1000  # number of steps for border search

# Plotting parameters
PLOT_XLIM_MIN = 0.2  # minimum x-axis limit for histogram
PLOT_XLIM_MAX = 2.5  # maximum x-axis limit for histogram
PLOT_BIN_STEP = 0.1  # histogram bin step size
PLOT_DPI = 300  # plot resolution

# Particle types
PARTICLE_TYPES = ["p", "N", "Fe"]  # proton, nitrogen, iron
PARTICLE_COLORS = ["r", "g", "b"]  # colors for plotting

# Input file paths (default values, can be overridden)
DEFAULT_INPUT_FILES = [
    "moshits_p_params.csv",
    "moshits_N_params.csv",
    "moshits_Fe_params.csv",
]

# Output file templates
OUTPUT_CRITERION_TEMPLATE = "criterion_Rc_{}_test.txt"
OUTPUT_PLOT_FILE = "criterion_Rc_all.pdf"

# Optimal parameters found from previous runs
OPTIMIZATION_RUN = True
OPTIMAL_R1_PN = 24
OPTIMAL_R2_PN = 174

# ============================================================================
# CORE COMPUTATIONAL FUNCTIONS
# ============================================================================


@numba.jit(nopython=True)
def integrand(r, p0, p1, p4, s):
    """
    Integrand function for LDF (Lateral Distribution Function) integration.

    Args:
        r: Radial distance from shower axis
        p0-p4, s: LDF shape parameters

    Returns:
        Integrand value at radius r
    """
    return 2.0 * np.pi * p0**2 * r / (((1 + p1 * r) ** 2) * (1 + p4 * r**s))


def integrate_function(a0, a1, a4, a5, r1, r2):
    """
    Integrate LDF over two radial regions: [0, r1] and [r1, r2].

    Args:
        a0-a5: LDF parameters (p0, p1, p2, p3, p4, s) as arrays
        r1: Inner radius boundary
        r2: Outer radius boundary

    Returns:
        Tuple (s1, s2): integrals over inner and outer regions
    """
    s1 = scipy.integrate.quad_vec(
        integrand,
        0,
        r1,
        args=(a0, a1, a4, a5),
        epsabs=INTEGRATION_ABS_ERROR,
        epsrel=INTEGRATION_REL_ERROR,
    )
    s2 = scipy.integrate.quad_vec(
        integrand,
        r1,
        r2,
        args=(a0, a1, a4, a5),
        epsabs=INTEGRATION_ABS_ERROR,
        epsrel=INTEGRATION_REL_ERROR,
    )
    return s1[0], s2[0]


def precompute_cumulative_tables(df_list, r_max, dr):
    """
    Precompute cumulative integrals for all events on a fixed radial grid.

    This turns many repeated quad calls into fast array lookups:
        S(r) = ∫0^r integrand(ρ) dρ  (trapezoid rule on a fine grid)

    Args:
        df_list: list of DataFrames (one per particle type)
        r_max: maximum radius to precompute (mm)
        dr: radial step (mm)

    Returns:
        tuple (r_grid, cum_tables, step_tables) where:
            r_grid: 1D ndarray of radii
            cum_tables: list of 2D arrays, one per particle type, shape (n_events, len(r_grid))
            step_tables: list of step arrays aligned with rows of cum_tables
    """
    r_grid = np.arange(0, r_max + dr, dr, dtype=np.float64)
    cum_tables = []
    step_tables = []
    for df in df_list:
        p0 = df["p0"].to_numpy(dtype=np.float64)
        p1 = df["p1"].to_numpy(dtype=np.float64)
        p4 = df["p4"].to_numpy(dtype=np.float64)
        s = df["s"].to_numpy(dtype=np.float64)
        steps = df["step"].to_numpy()

        # Broadcast over events (rows) and radii (cols)
        r = r_grid[None, :]
        numer = 2.0 * np.pi * (p0[:, None] ** 2) * r
        denom = ((1.0 + p1[:, None] * r) ** 2) * (1.0 + p4[:, None] * (r ** s[:, None]))
        integrand_vals = numer / denom

        # Cumulative integral using rectangle rule (stable and fast)
        cum = np.cumsum(integrand_vals * dr, axis=1)
        cum_tables.append(cum)
        step_tables.append(steps)
    return r_grid, cum_tables, step_tables


def compute_criteria(df, r1, r2, anomaly_threshold=DEFAULT_ANOMALY_THRESHOLD):
    """
    Compute criterion values for a single particle type dataset.

    Args:
        df: DataFrame with LDF parameters (p0, p1, p2, p3, p4, s)
        r1: Inner radius boundary
        r2: Outer radius boundary
        anomaly_threshold: Threshold for filtering anomalous values

    Returns:
        Tuple (average, filtered_criterion, event_names):
            - average: Mean criterion value after filtering
            - filtered_criterion: Array of criterion values without anomalies
            - event_names: Step numbers of non-anomalous events
    """
    a0, a1, a4, a5 = (df["p0"].values, df["p1"].values, df["p4"].values, df["s"].values)
    s1, s2 = integrate_function(a0, a1, a4, a5, r1, r2)

    # Avoid division by zero and infinity
    with np.errstate(divide="ignore", invalid="ignore"):
        cri_j = np.divide(s1, s2, out=np.zeros_like(s1), where=s2 != 0)

    # Filter out anomalies, NaN, and inf values
    valid_mask = np.isfinite(cri_j) & (cri_j <= anomaly_threshold) & (cri_j > 0)
    filtered_cri_j = cri_j[valid_mask]
    event_names = df["step"].to_numpy()[valid_mask]

    if filtered_cri_j.size > 0:
        average = np.mean(filtered_cri_j)
    else:
        average = float("nan")
        logger.warning(f"No valid criterion values for r1={r1}, r2={r2}")

    return average, filtered_cri_j, event_names


def compute_criteria_fast(
    r1, r2, cum_table, r_grid, steps, anomaly_threshold=DEFAULT_ANOMALY_THRESHOLD
):
    """
    Fast criterion computation using precomputed cumulative integrals.
    """
    # find nearest indices on the grid
    i1 = int(np.searchsorted(r_grid, r1, side="left"))
    i2 = int(np.searchsorted(r_grid, r2, side="left"))
    i1 = min(max(i1, 0), len(r_grid) - 1)
    i2 = min(max(i2, i1 + 1), len(r_grid) - 1)

    s1 = cum_table[:, i1]
    s2 = cum_table[:, i2] - cum_table[:, i1]

    with np.errstate(divide="ignore", invalid="ignore"):
        cri_j = np.divide(s1, s2, out=np.zeros_like(s1), where=s2 != 0)

    valid_mask = np.isfinite(cri_j) & (cri_j <= anomaly_threshold) & (cri_j > 0)
    filtered_cri_j = cri_j[valid_mask]
    event_names = steps[valid_mask]

    if filtered_cri_j.size > 0:
        average = float(np.mean(filtered_cri_j))
    else:
        average = float("nan")
        logger.warning(f"No valid criterion values for r1={r1}, r2={r2} (fast)")

    return average, filtered_cri_j, event_names


def criterion(
    df_list,
    r1,
    r2,
    anomaly_threshold=DEFAULT_ANOMALY_THRESHOLD,
    cum_tables=None,
    r_grid=None,
    steps=None,
):
    """
    Compute criterion values for multiple particle type datasets.

    Args:
        df_list: List of DataFrames, one per particle type
        r1: Inner radius boundary
        r2: Outer radius boundary
        anomaly_threshold: Threshold for filtering anomalous values

    Returns:
        Tuple (averages, criteria, event_names):
            - averages: List of mean criterion values per particle type
            - criteria: List of filtered criterion arrays per particle type
            - event_names: List of event name arrays per particle type
    """
    averages = []
    criteria = []
    event_names = []

    for idx, df in enumerate(df_list):
        if (
            cum_tables is not None
            and r_grid is not None
            and steps is not None
        ):
            average, cri_j, names = compute_criteria_fast(
                r1, r2, cum_tables[idx], r_grid, steps[idx], anomaly_threshold
            )
        else:
            average, cri_j, names = compute_criteria(df, r1, r2, anomaly_threshold)
        averages.append(average)
        criteria.append(cri_j)
        event_names.append(names)

    return averages, criteria, event_names


def sep_two(data):
    """
    Find optimal separation boundary between two particle populations.

    Searches for boundary that minimizes maximum misclassification error
    between two datasets.

    Args:
        data: List/tuple of two arrays containing criterion values

    Returns:
        List [min_error, optimal_boundary]:
            - min_error: Minimum maximum error achieved
            - optimal_boundary: Border value that achieves min_error
    """
    if len(data) < 2:
        logger.error("sep_two requires at least 2 datasets")
        return [float("nan"), float("nan")]

    min_max_error = float("inf")
    optimal_border = None
    border_step = (SEP_BORDER_MAX - SEP_BORDER_MIN) / SEP_BORDER_STEPS

    for border in np.arange(SEP_BORDER_MIN, SEP_BORDER_MAX, border_step):
        if len(data[0]) == 0 or len(data[1]) == 0:
            continue

        # Fraction misclassified in first dataset (should be < border)
        error_first = np.count_nonzero(data[0] < border) / float(len(data[0]))
        # Fraction misclassified in second dataset (should be > border)
        error_second = np.count_nonzero(data[1] > border) / float(len(data[1]))

        max_error = max(error_first, error_second)
        if max_error < min_max_error:
            min_max_error = max_error
            optimal_border = border

    if optimal_border is not None:
        return [round(min_max_error, 3), round(optimal_border, 3)]
    else:
        logger.warning("Could not find optimal separation boundary")
        return [float("nan"), float("nan")]


def compute_error(r1, r2, data_list, cum_tables=None, r_grid=None, steps=None):
    """
    Compute classification errors for given radius boundaries.

    Args:
        r1: Inner radius boundary
        r2: Outer radius boundary
        data_list: List of DataFrames for different particle types

    Returns:
        Tuple (r1, r2, error_pn, error_nfe):
            - r1, r2: Input radius boundaries
            - error_pn: Classification error for proton-nitrogen separation
            - error_nfe: Classification error for nitrogen-iron separation
    """
    aver, cri, _ = criterion(
        data_list, r1, r2, cum_tables=cum_tables, r_grid=r_grid, steps=steps
    )
    res_border_pn = sep_two(cri[:2])  # p vs N
    res_border_nfe = sep_two(cri[1:])  # N vs Fe
    return r1, r2, res_border_pn[0], res_border_nfe[0]


def select_optimal_radii(res):
    """
    Select optimal radii from optimization results.

    Args:
        res: List of tuples (r1, r2, error_pn, error_nfe)

    Returns:
        Dictionary with optimization strategies:
            - 'minimax': (r1, r2, err_pn, err_nfe) minimizing max(err_pn, err_nfe)
            - 'minsum': (r1, r2, err_pn, err_nfe) minimizing total error
            - 'pareto': List of non-dominated (r1, r2, err_pn, err_nfe) points
    """
    if not res:
        logger.warning("No optimization results provided")
        return {"minimax": None, "minsum": None, "pareto": []}

    # Convert to numpy array and filter out NaN/inf values
    arr = np.array(res, dtype=float)  # shape (N, 4)
    errs = arr[:, 2:4]

    # Filter out rows with NaN or inf in error columns
    valid_mask = np.all(np.isfinite(errs), axis=1)
    if not np.any(valid_mask):
        logger.error("All optimization results contain NaN or inf values")
        return {"minimax": None, "minsum": None, "pareto": []}

    arr_valid = arr[valid_mask]
    errs_valid = arr_valid[:, 2:4]

    # Strategy 1: Minimax - minimize maximum error
    max_err = np.max(errs_valid, axis=1)
    idx_minimax = int(np.argmin(max_err))
    minimax_result = tuple(arr_valid[idx_minimax].tolist())

    # Strategy 2: Minsum - minimize sum of errors
    sum_err = np.sum(errs_valid, axis=1)
    idx_minsum = int(np.argmin(sum_err))
    minsum_result = tuple(arr_valid[idx_minsum].tolist())

    # Strategy 3: Pareto frontier - non-dominated points
    pareto_results = []
    N = arr_valid.shape[0]
    for i in range(N):
        e_i = errs_valid[i]
        dominated = False
        for j in range(N):
            if i == j:
                continue
            e_j = errs_valid[j]
            # Point j dominates point i if j is better or equal in both dimensions
            # and strictly better in at least one dimension
            if (e_j[0] <= e_i[0] and e_j[1] <= e_i[1]) and (
                e_j[0] < e_i[0] or e_j[1] < e_i[1]
            ):
                dominated = True
                break
        if not dominated:
            pareto_results.append(tuple(arr_valid[i].tolist()))

    logger.info(
        f"Found {len(arr_valid)} valid optimization results "
        f"({len(arr) - len(arr_valid)} filtered due to NaN/inf)"
    )
    logger.info(f"Pareto frontier contains {len(pareto_results)} points")

    return {
        "minimax": minimax_result,
        "minsum": minsum_result,
        "pareto": pareto_results,
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("COSMIC RAY PARTICLE CLASSIFICATION CRITERION OPTIMIZATION")
    logger.info("=" * 70)

    # Load input data
    logger.info("Loading particle data files...")
    try:
        data_list = [pd.read_csv(file) for file in DEFAULT_INPUT_FILES]
        logger.info(f"Loaded {len(data_list)} particle type datasets")
        for i, (df, ptype) in enumerate(zip(data_list, PARTICLE_TYPES)):
            logger.info(f"  {ptype}: {len(df)} events")
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Initialize tracking variables for optimization
    min_error_pn = float("inf")
    min_error_nfe = float("inf")
    min_r1_pn, min_r2_pn = None, None
    min_r1_nfe, min_r2_nfe = None, None

    # Generate radius ranges
    r1_range = np.arange(R1_MIN, R1_MAX, R1_STEP)
    r2_range = np.arange(R2_MIN, R2_MAX, R2_STEP)
    total_combinations = len(r1_range) * len(r2_range)

    logger.info(
        f"Starting parameter optimization over {total_combinations} combinations"
    )
    logger.info(f"  r1 range: [{R1_MIN}, {R1_MAX}), step={R1_STEP}")
    logger.info(f"  r2 range: [{R2_MIN}, {R2_MAX}), step={R2_STEP}")

    if OPTIMIZATION_RUN:
        # Precompute cumulative integrals once and reuse for all (r1, r2)
        logger.info("Precomputing cumulative integrals on radial grid...")
        r_grid, cum_tables, step_tables = precompute_cumulative_tables(
            data_list, r_max=R2_MAX, dr=RADIAL_STEP
        )

        results = []
        processed = 0
        for r1 in r1_range:
            for r2 in r2_range:
                r1, r2, error_pn, error_nfe = compute_error(
                    r1, r2, data_list, cum_tables=cum_tables, r_grid=r_grid, steps=step_tables
                )
                results.append((r1, r2, error_pn, error_nfe))
                processed += 1
                if processed % 200 == 0:
                    logger.info(
                        f"Processed {processed}/{total_combinations} combinations"
                    )
        # Select optimal radii using different strategies
        choices = select_optimal_radii(results)
        minimax = choices.get("minimax")
        minsum = choices.get("minsum")
        pareto = choices.get("pareto", [])

        # Update optimal parameters if minimax strategy found a solution
        if minimax is not None:
            opt_r1, opt_r2, opt_err_pn, opt_err_nfe = minimax
            OPTIMAL_R1_PN = int(opt_r1)
            OPTIMAL_R2_PN = int(opt_r2)
            logger.info("Optimization complete!")
            logger.info(
                "minimax -> r1=%d r2=%d err_pn=%.3f err_nfe=%.3f",
                int(opt_r1),
                int(opt_r2),
                opt_err_pn,
                opt_err_nfe,
            )
        else:
            logger.info("Optimization complete!")
            logger.info("minimax: None (using default values)")

        if minsum is not None:
            logger.info(
                "minsum -> r1=%d r2=%d err_pn=%.3f err_nfe=%.3f",
                int(minsum[0]),
                int(minsum[1]),
                minsum[2],
                minsum[3],
            )
        else:
            logger.info("minsum: None")

        if pareto:
            logger.info("pareto frontier (r1, r2, err_pn, err_nfe):")
            for p in pareto:
                logger.info(
                    "  r1=%d r2=%d err_pn=%.3f err_nfe=%.3f",
                    int(p[0]),
                    int(p[1]),
                    p[2],
                    p[3],
                )
        else:
            logger.info("pareto frontier: empty")
    # Ensure precomputed tables exist for the final criterion calculation
    if "step_tables" not in locals():
        r_grid, cum_tables, step_tables = precompute_cumulative_tables(
            data_list, r_max=R2_MAX, dr=RADIAL_STEP
        )

    # Compute criterion values using optimal parameters
    logger.info(f"Computing criterion with r1={OPTIMAL_R1_PN}, r2={OPTIMAL_R2_PN}")
    aver, cri, event_names = criterion(
        data_list,
        OPTIMAL_R1_PN,
        OPTIMAL_R2_PN,
        cum_tables=cum_tables,
        r_grid=r_grid,
        steps=step_tables,
    )
    res_border_pn = sep_two(cri[:2])
    res_border_nfe = sep_two(cri[1:])
    logger.info(f"Separation borders: p-N={res_border_pn}, N-Fe={res_border_nfe}")

    # Save criterion values to files
    logger.info("Saving criterion values to files...")
    for i, (particle_type, criterion_values) in enumerate(zip(PARTICLE_TYPES, cri)):
        filename = OUTPUT_CRITERION_TEMPLATE.format(particle_type)
        df_output = pd.DataFrame({"name": event_names[i], "cri": criterion_values})
        df_output.to_csv(filename, index=False)
        logger.info(f"  Saved {filename}")

    # Generate histogram plot
    logger.info("Generating histogram plot...")
    min_vals = [min(c) for c in cri]
    max_vals = [max(c) for c in cri]
    min_r, max_r = min(min_vals), max(max_vals)
    bins = np.arange(min_r, max_r, PLOT_BIN_STEP)

    fig = plt.figure()
    for i in range(len(cri)):
        plt.hist(
            cri[i], bins, alpha=0.3, label=PARTICLE_TYPES[i], color=PARTICLE_COLORS[i]
        )

    plt.ylabel(f"count in bin {PLOT_BIN_STEP}")
    plt.xlabel("R")
    plt.xlim([PLOT_XLIM_MIN, PLOT_XLIM_MAX])
    plt.axvline(res_border_pn[1], color="r", linewidth=1)
    plt.axvline(res_border_nfe[1], color="b", linewidth=1)
    plt.legend()
    plt.title("Распределение критериального параметра Rc по ядрам")
    plt.savefig(OUTPUT_PLOT_FILE, bbox_inches="tight", dpi=PLOT_DPI)
    logger.info(f"Saved plot to {OUTPUT_PLOT_FILE}")

    logger.info("=" * 70)
    logger.info("ALL PROCESSING COMPLETE")
    logger.info("=" * 70)
