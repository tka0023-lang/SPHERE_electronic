# sphere_appro/criterion.py
import logging
from pathlib import Path

import numpy as np

from .config import (
    R1_MIN, R1_MAX, R1_STEP, R2_MIN, R2_MAX, R2_STEP,
    RADIAL_STEP, SEP_BORDER_MIN, SEP_BORDER_MAX, SEP_BORDER_STEPS,
    DEFAULT_ANOMALY_THRESHOLD, RESULT_COLUMNS,
)
from .io_data import load_results_csv, save_results_csv

try:
    from .ldf_core import ldf_integrand
except ImportError:
    from ._ldf_core_fallback import ldf_integrand

logger = logging.getLogger(__name__)

PARTICLE_TYPES = ['p', 'N', 'Fe']
DEFAULT_INPUT_FILES = [f'moshits_{t}_params.csv' for t in PARTICLE_TYPES]


def find_optimal_border(data_a, data_b, n_steps=SEP_BORDER_STEPS):
    """Vectorized optimal border search using np.searchsorted.
    data_a: values expected ABOVE the border (e.g. protons).
    data_b: values expected BELOW the border (e.g. nitrogen).
    err_a = fraction(a < border) = misclassified fraction of a.
    err_b = fraction(b > border) = misclassified fraction of b.
    Returns (min_max_error, optimal_border)."""
    borders = np.linspace(SEP_BORDER_MIN, SEP_BORDER_MAX, n_steps)
    sorted_a = np.sort(data_a)
    sorted_b = np.sort(data_b)
    err_a = np.searchsorted(sorted_a, borders) / len(data_a)
    err_b = 1.0 - np.searchsorted(sorted_b, borders) / len(data_b)
    max_err = np.maximum(err_a, err_b)
    idx = int(np.argmin(max_err))
    return float(max_err[idx]), float(borders[idx])


def precompute_cumulative_tables(p0, p1, p2, p3, p4, p5, p6, R_ch, sw, r_max, dr):
    """Precompute cumulative integrals on a radial grid using F_new model.
    Args: p0..sw are 1D numpy arrays (one per event).
    Returns: (r_grid, cum_table) where cum_table is (n_events, n_radii)."""
    r_grid = np.arange(0, r_max + dr, dr, dtype=np.float64)
    n_events = len(p0)
    n_radii = len(r_grid)
    integrand_vals = np.empty((n_events, n_radii), dtype=np.float64)
    for i in range(n_events):
        for j, r in enumerate(r_grid):
            integrand_vals[i, j] = ldf_integrand(
                r, p0[i], p1[i], p2[i], p3[i],
                p4[i], p5[i], p6[i], R_ch[i], sw[i],
            )
    cum = np.cumsum(integrand_vals * dr, axis=1)
    return r_grid, cum


def compute_criteria_fast(r1, r2, cum_table, r_grid,
                          anomaly_threshold=DEFAULT_ANOMALY_THRESHOLD):
    """Compute criterion S(0..r1)/S(r1..r2) from cumulative tables."""
    i1 = min(max(int(np.searchsorted(r_grid, r1)), 0), len(r_grid) - 1)
    i2 = min(max(int(np.searchsorted(r_grid, r2)), i1 + 1), len(r_grid) - 1)
    s1 = cum_table[:, i1]
    s2 = cum_table[:, i2] - cum_table[:, i1]
    with np.errstate(divide='ignore', invalid='ignore'):
        cri = np.divide(s1, s2, out=np.zeros_like(s1), where=s2 != 0)
    valid = np.isfinite(cri) & (cri <= anomaly_threshold) & (cri > 0)
    return cri, valid


def compute_error(r1, r2, cum_tables, r_grid):
    """Classification errors for p-N and N-Fe at given (r1, r2)."""
    criteria = []
    for cum in cum_tables:
        cri, valid = compute_criteria_fast(r1, r2, cum, r_grid)
        criteria.append(cri[valid])
    if len(criteria) < 3 or any(len(c) == 0 for c in criteria):
        return r1, r2, float('nan'), float('nan')
    err_pn, _ = find_optimal_border(criteria[0], criteria[1])
    err_nfe, _ = find_optimal_border(criteria[1], criteria[2])
    return r1, r2, err_pn, err_nfe


def select_optimal_radii(results):
    """Select optimal (r1,r2) via minimax, minsum, Pareto."""
    arr = np.array(results, dtype=float)
    errs = arr[:, 2:4]
    valid_mask = np.all(np.isfinite(errs), axis=1)
    if not np.any(valid_mask):
        return {'minimax': None, 'minsum': None, 'pareto': []}
    arr_v = arr[valid_mask]
    errs_v = arr_v[:, 2:4]
    idx_mm = int(np.argmin(np.max(errs_v, axis=1)))
    idx_ms = int(np.argmin(np.sum(errs_v, axis=1)))
    pareto = []
    for i in range(len(arr_v)):
        dominated = False
        for j in range(len(arr_v)):
            if i != j and np.all(errs_v[j] <= errs_v[i]) and np.any(errs_v[j] < errs_v[i]):
                dominated = True
                break
        if not dominated:
            pareto.append(tuple(arr_v[i]))
    return {
        'minimax': tuple(arr_v[idx_mm]),
        'minsum': tuple(arr_v[idx_ms]),
        'pareto': pareto,
    }


def optimize_radii(cum_tables, r_grid):
    """Grid search over (r1, r2) combinations."""
    r1_range = np.arange(R1_MIN, R1_MAX, R1_STEP)
    r2_range = np.arange(R2_MIN, R2_MAX, R2_STEP)
    results = []
    for r1 in r1_range:
        for r2 in r2_range:
            results.append(compute_error(r1, r2, cum_tables, r_grid))
    return select_optimal_radii(results)


def run_criterion(input_files=None, output_dir='.'):
    """Main criterion pipeline: load CSVs, optimize radii, save criterion files."""
    if input_files is None:
        input_files = DEFAULT_INPUT_FILES
    data_list = []
    col_idx = {}
    for f in input_files:
        data, col_idx = load_results_csv(f)
        data_list.append(data)
    if not data_list:
        logger.warning('No input files provided')
        return {'minimax': None, 'minsum': None, 'pareto': []}

    idx = {name: col_idx[name] for name in ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'R_ch', 'sw']}

    cum_tables = []
    for data in data_list:
        _, cum = precompute_cumulative_tables(
            data[:, idx['p0']], data[:, idx['p1']],
            data[:, idx['p2']], data[:, idx['p3']],
            data[:, idx['p4']], data[:, idx['p5']],
            data[:, idx['p6']], data[:, idx['R_ch']],
            data[:, idx['sw']],
            r_max=R2_MAX, dr=RADIAL_STEP,
        )
        cum_tables.append(cum)

    r_grid = np.arange(0, R2_MAX + RADIAL_STEP, RADIAL_STEP)

    choices = optimize_radii(cum_tables, r_grid)
    logger.info('Optimization results: %s', choices)

    if choices['minimax'] is not None:
        opt_r1, opt_r2 = int(choices['minimax'][0]), int(choices['minimax'][1])
    else:
        opt_r1, opt_r2 = 24, 174

    for i, ptype in enumerate(PARTICLE_TYPES):
        cri, valid = compute_criteria_fast(opt_r1, opt_r2, cum_tables[i], r_grid)
        step_col = col_idx.get('step', None)
        if step_col is not None:
            names = data_list[i][valid, step_col]
        else:
            names = np.arange(len(cri))[valid].astype(float)
        filename = f'{output_dir}/criterion_Rc_{ptype}_test.txt'
        with open(filename, 'w') as f:
            f.write('name,cri\n')
            for name, c in zip(names, cri[valid]):
                f.write(f'{int(name)},{c:.6f}\n')
        logger.info('Saved %s (%d values)', filename, valid.sum())

    return choices


def run_criterion_parquet(parquet_path, output_dir='.', min_events_per_class=50):
    """Run criterion analysis on a parquet file with particle/energy/angle/height columns."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    # Criterion only needs shape params (p0-p6, R_ch, sw); drop rows with NaN/inf in those
    shape_cols = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'R_ch', 'sw']
    mask = df[shape_cols].isna().any(axis=1) | np.isinf(df[shape_cols].values).any(axis=1)
    if mask.any():
        logger.info('Dropping %d rows with NaN/inf in shape params', mask.sum())
        df = df[~mask].reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_rows = []

    def _compute_for_subset(subset_df, label):
        particle_groups = {}
        for ptype in ['p', 'N', 'Fe']:
            grp = subset_df[subset_df['particle'] == ptype]
            if len(grp) < min_events_per_class:
                logger.warning('Slice %s: particle %s has only %d events (< %d), skipping',
                               label, ptype, len(grp), min_events_per_class)
                return None
            particle_groups[ptype] = grp

        cum_tables = []
        r_grid = None
        for ptype in ['p', 'N', 'Fe']:
            grp = particle_groups[ptype]
            rg, ct = precompute_cumulative_tables(
                grp['p0'].values, grp['p1'].values,
                grp['p2'].values, grp['p3'].values,
                grp['p4'].values, grp['p5'].values,
                grp['p6'].values, grp['R_ch'].values,
                grp['sw'].values,
                r_max=R2_MAX, dr=RADIAL_STEP,
            )
            cum_tables.append(ct)
            if r_grid is None:
                r_grid = rg

        choices = optimize_radii(cum_tables, r_grid)
        minimax = choices.get('minimax')
        if minimax is None:
            logger.warning('Slice %s: optimize_radii returned None for minimax', label)
            return None

        opt_r1, opt_r2, err_pn, err_nfe = minimax

        cri_p, valid_p = compute_criteria_fast(opt_r1, opt_r2, cum_tables[0], r_grid)
        cri_n, valid_n = compute_criteria_fast(opt_r1, opt_r2, cum_tables[1], r_grid)
        cri_fe, valid_fe = compute_criteria_fast(opt_r1, opt_r2, cum_tables[2], r_grid)

        _, border_pn = find_optimal_border(cri_p[valid_p], cri_n[valid_n])
        _, border_nfe = find_optimal_border(cri_n[valid_n], cri_fe[valid_fe])

        return {
            'r1_opt': opt_r1, 'r2_opt': opt_r2,
            'error_pN': err_pn, 'error_NFe': err_nfe,
            'border_pN': border_pn, 'border_NFe': border_nfe,
        }

    full_result = _compute_for_subset(df, 'full')
    if full_result is not None:
        results_rows.append({'energy': 'all', 'angle': 0, 'height': 0, **full_result})

    for (energy, angle, height), slice_df in df.groupby(['energy', 'angle', 'height']):
        label = f'{energy}/{angle}/{height}'
        row = _compute_for_subset(slice_df, label)
        if row is not None:
            results_rows.append({'energy': energy, 'angle': angle, 'height': height, **row})

    result_df = pd.DataFrame(results_rows)
    out_path = output_dir / 'criterion_results.parquet'
    result_df.to_parquet(out_path, index=False)
    logger.info('Criterion results saved to %s (%d rows)', out_path, len(result_df))
    return str(out_path)
