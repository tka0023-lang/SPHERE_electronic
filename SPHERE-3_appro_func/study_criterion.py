#!/usr/bin/env python3
"""Multi-ring criterion study: compare classification variants across datasets."""
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sphere_appro.criterion import (
    precompute_cumulative_tables, compute_criteria_fast,
    find_optimal_border,
)
from sphere_appro.config import RADIAL_STEP

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ATM_FILES = {
    'atm00_sb': Path('results_sb.parquet'),
    'atm01': Path('results_fnew2.parquet'),
    'atm03': Path('../results/results_111.parquet'),
    'atm04': Path('../results/results_50.parquet'),
}
ATM_LABELS = list(ATM_FILES.keys())
OUTPUT_DIR = Path('analysis_output_criterion_study')
PLOT_DIR = OUTPUT_DIR / 'plots'

SHAPE_PARAMS = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'R_ch', 'sw']
PARTICLES = ['p', 'N', 'Fe']

R_MAX = 270
MIN_ZONE_WIDTH = 20
COARSE_STEP = 10
FINE_STEP = 2
FINE_NEIGHBORHOOD = 15

VARIANT_NAMES = ['baseline', '2ring_fisher', '3ring_fisher', '2ring_minimax']


def load_data() -> pd.DataFrame:
    """Read all parquet files, tag with 'atm' column, concatenate."""
    frames = []
    for label, path in ATM_FILES.items():
        logger.info('Loading %s from %s ...', label, path)
        df = pd.read_parquet(path)
        df['atm'] = label
        frames.append(df)
        logger.info('  %s: %d rows', label, len(df))
    combined = pd.concat(frames, ignore_index=True)
    logger.info('Combined dataset: %d rows', len(combined))
    return combined


def build_cum_tables(df: pd.DataFrame) -> dict:
    """Build cumulative tables keyed by (atm, particle) -> (r_grid, cum_table)."""
    tables = {}
    for atm in ATM_LABELS:
        for particle in PARTICLES:
            sub = df[(df['atm'] == atm) & (df['particle'] == particle)]
            if len(sub) == 0:
                logger.warning('No data for atm=%s particle=%s', atm, particle)
                continue
            mask = sub[SHAPE_PARAMS].isna().any(axis=1) | np.isinf(sub[SHAPE_PARAMS].values).any(axis=1)
            sub = sub[~mask]
            if len(sub) == 0:
                continue
            r_grid, cum = precompute_cumulative_tables(
                sub['p0'].values, sub['p1'].values,
                sub['p2'].values, sub['p3'].values,
                sub['p4'].values, sub['p5'].values,
                sub['p6'].values, sub['R_ch'].values,
                sub['sw'].values,
                r_max=R_MAX, dr=RADIAL_STEP,
            )
            tables[(atm, particle)] = (r_grid, cum)
            logger.info('  cum table (%s, %s): %d events, grid %d',
                        atm, particle, cum.shape[0], cum.shape[1])
    return tables


# ===================================================================
# Zone features and classifiers
# ===================================================================

def compute_zone_features(cum_table: np.ndarray, r_grid: np.ndarray,
                          radii: list[float]) -> np.ndarray:
    """Compute log-ratio features from cumulative table for given zone radii.

    Args:
        cum_table: (n_events, n_radii) cumulative integral table
        r_grid: radial grid values
        radii: list of N radii [r1, r2, ..., rN] defining N zones

    Returns:
        features: (n_events, N-1) array of log(Zk / Z_{k+1})
        valid: boolean mask of events with finite features
    """
    indices = [min(max(int(np.searchsorted(r_grid, r)), 0), len(r_grid) - 1) for r in radii]

    # Zone integrals: Z1 = S(r1), Z2 = S(r2) - S(r1), ...
    zones = []
    zones.append(cum_table[:, indices[0]])  # Z1 = S(0..r1)
    for k in range(1, len(indices)):
        zones.append(cum_table[:, indices[k]] - cum_table[:, indices[k - 1]])

    n_features = len(zones) - 1
    features = np.empty((cum_table.shape[0], n_features), dtype=np.float64)

    for k in range(n_features):
        with np.errstate(divide='ignore', invalid='ignore'):
            features[:, k] = np.log(np.divide(
                zones[k], zones[k + 1],
                out=np.zeros(cum_table.shape[0]),
                where=zones[k + 1] > 0,
            ))

    valid = np.all(np.isfinite(features), axis=1)
    return features, valid


def find_optimal_border_adaptive(data_a: np.ndarray, data_b: np.ndarray,
                                  n_steps: int = 1000):
    """Find optimal border with data-adaptive range.

    Like find_optimal_border but determines border search range from data.
    data_a: values expected ABOVE the border.
    data_b: values expected BELOW the border.
    """
    all_vals = np.concatenate([data_a, data_b])
    lo = np.percentile(all_vals, 1)
    hi = np.percentile(all_vals, 99)
    margin = (hi - lo) * 0.1
    borders = np.linspace(lo - margin, hi + margin, n_steps)

    sorted_a = np.sort(data_a)
    sorted_b = np.sort(data_b)
    err_a = np.searchsorted(sorted_a, borders) / len(data_a)
    err_b = 1.0 - np.searchsorted(sorted_b, borders) / len(data_b)
    max_err = np.maximum(err_a, err_b)
    idx = int(np.argmin(max_err))
    return float(max_err[idx]), float(borders[idx])


def fisher_lda(features_a: np.ndarray, features_b: np.ndarray):
    """Compute Fisher LDA direction and project both classes.

    Args:
        features_a: (n_a, d) features for class a (expected above border)
        features_b: (n_b, d) features for class b (expected below border)

    Returns:
        w: (d,) Fisher direction (unit vector)
        scores_a: (n_a,) projections of class a onto w
        scores_b: (n_b,) projections of class b onto w
    """
    mu_a = np.mean(features_a, axis=0)
    mu_b = np.mean(features_b, axis=0)

    # Within-class scatter (pooled covariance)
    diff_a = features_a - mu_a
    diff_b = features_b - mu_b
    S_w = (diff_a.T @ diff_a + diff_b.T @ diff_b) / (len(features_a) + len(features_b))

    # Fisher direction
    try:
        w = np.linalg.solve(S_w, mu_a - mu_b)
    except np.linalg.LinAlgError:
        w = mu_a - mu_b  # fallback if singular

    norm = np.linalg.norm(w)
    if norm > 0:
        w = w / norm

    scores_a = features_a @ w
    scores_b = features_b @ w

    return w, scores_a, scores_b


def minimax_2d(features_a: np.ndarray, features_b: np.ndarray,
               n_angles: int = 180):
    """Find optimal linear boundary in 2D feature space via minimax.

    Searches over boundary angles theta in [0, pi).
    For each angle, projects onto the direction and uses find_optimal_border_adaptive
    to find the best threshold.
    Boundary: cos(theta)*f1 + sin(theta)*f2 = c
    Class a is expected on the positive side.

    Returns:
        w: (2,) normal direction of optimal boundary
        scores_a, scores_b: projections onto w
        best_err: minimax error at optimal boundary
    """
    best_err = 1.0
    best_w = np.array([1.0, 0.0])

    for theta in np.linspace(0, np.pi, n_angles, endpoint=False):
        w = np.array([np.cos(theta), np.sin(theta)])
        proj_a = features_a @ w
        proj_b = features_b @ w

        # Check if class a should be above or below
        if np.mean(proj_a) < np.mean(proj_b):
            proj_a, proj_b = proj_b, proj_a
            w = -w

        err, _ = find_optimal_border_adaptive(proj_a, proj_b)
        if err < best_err:
            best_err = err
            best_w = w

    # Final projection with best direction
    scores_a = features_a @ best_w
    scores_b = features_b @ best_w

    return best_w, scores_a, scores_b, best_err


# ===================================================================
# Radius optimization and variant training
# ===================================================================

def _generate_radii_grid(n_radii: int, step: int,
                         bounds: list[tuple[int, int]] | None = None) -> list[list[int]]:
    """Generate all valid (r1, ..., rN) combinations with min zone width constraint."""
    if bounds is None:
        if n_radii == 2:
            bounds = [(50, 110), (110, 270)]
        elif n_radii == 3:
            bounds = [(30, 100), (50, 160), (70, 270)]
        elif n_radii == 4:
            bounds = [(30, 90), (50, 140), (70, 210), (90, 270)]
        else:
            raise ValueError(f'Unsupported n_radii={n_radii}')

    ranges = [np.arange(lo, hi + 1, step) for lo, hi in bounds]

    combos = []
    if n_radii == 2:
        for r1 in ranges[0]:
            for r2 in ranges[1]:
                if r2 >= r1 + MIN_ZONE_WIDTH:
                    combos.append([int(r1), int(r2)])
    elif n_radii == 3:
        for r1 in ranges[0]:
            for r2 in ranges[1]:
                if r2 < r1 + MIN_ZONE_WIDTH:
                    continue
                for r3 in ranges[2]:
                    if r3 >= r2 + MIN_ZONE_WIDTH:
                        combos.append([int(r1), int(r2), int(r3)])
    elif n_radii == 4:
        for r1 in ranges[0]:
            for r2 in ranges[1]:
                if r2 < r1 + MIN_ZONE_WIDTH:
                    continue
                for r3 in ranges[2]:
                    if r3 < r2 + MIN_ZONE_WIDTH:
                        continue
                    for r4 in ranges[3]:
                        if r4 >= r3 + MIN_ZONE_WIDTH:
                            combos.append([int(r1), int(r2), int(r3), int(r4)])
    return combos


def _eval_fisher_at_radii(radii, cum_tables, r_grid):
    """Evaluate Fisher criterion at given radii. Returns (err_pN, err_NFe, trained_model)."""
    feats_by_particle = {}
    for i, particle in enumerate(PARTICLES):
        feats, valid = compute_zone_features(cum_tables[i], r_grid, radii)
        feats_by_particle[particle] = feats[valid]

    if any(len(f) < 50 for f in feats_by_particle.values()):
        return float('nan'), float('nan'), None

    # p-N Fisher
    w_pn, scores_p_pn, scores_n_pn = fisher_lda(feats_by_particle['p'], feats_by_particle['N'])
    err_pn, border_pn = find_optimal_border_adaptive(scores_p_pn, scores_n_pn)

    # N-Fe Fisher
    w_nfe, scores_n_nfe, scores_fe_nfe = fisher_lda(feats_by_particle['N'], feats_by_particle['Fe'])
    err_nfe, border_nfe = find_optimal_border_adaptive(scores_n_nfe, scores_fe_nfe)

    model = {
        'radii': radii,
        'w_pN': w_pn, 'border_pN': border_pn,
        'w_NFe': w_nfe, 'border_NFe': border_nfe,
        'err_pN': err_pn, 'err_NFe': err_nfe,
    }
    return err_pn, err_nfe, model


def _eval_minimax2d_at_radii(radii, cum_tables, r_grid):
    """Evaluate minimax 2D criterion at given radii. Returns (err_pN, err_NFe, trained_model)."""
    feats_by_particle = {}
    for i, particle in enumerate(PARTICLES):
        feats, valid = compute_zone_features(cum_tables[i], r_grid, radii)
        feats_by_particle[particle] = feats[valid]

    if any(len(f) < 50 for f in feats_by_particle.values()):
        return float('nan'), float('nan'), None

    w_pn, _, _, err_pn = minimax_2d(feats_by_particle['p'], feats_by_particle['N'])
    scores_p_pn = feats_by_particle['p'] @ w_pn
    scores_n_pn = feats_by_particle['N'] @ w_pn
    _, border_pn = find_optimal_border_adaptive(scores_p_pn, scores_n_pn)

    w_nfe, _, _, err_nfe = minimax_2d(feats_by_particle['N'], feats_by_particle['Fe'])
    scores_n_nfe = feats_by_particle['N'] @ w_nfe
    scores_fe_nfe = feats_by_particle['Fe'] @ w_nfe
    _, border_nfe = find_optimal_border_adaptive(scores_n_nfe, scores_fe_nfe)

    model = {
        'radii': radii,
        'w_pN': w_pn, 'border_pN': border_pn,
        'w_NFe': w_nfe, 'border_NFe': border_nfe,
        'err_pN': err_pn, 'err_NFe': err_nfe,
    }
    return err_pn, err_nfe, model


def optimize_radii_multi(cum_tables, r_grid, n_radii, use_minimax_2d=False):
    """Two-stage grid search for optimal radii (3 or 4 radii).

    Args:
        cum_tables: list of 3 cumulative tables [p, N, Fe]
        r_grid: radial grid
        n_radii: 2, 3, or 4
        use_minimax_2d: if True, use minimax_2d instead of Fisher for 2D features

    Returns:
        best_model: dict with radii, weights, borders, errors
    """
    eval_fn = _eval_minimax2d_at_radii if use_minimax_2d else _eval_fisher_at_radii

    # Stage 1: coarse grid
    coarse_combos = _generate_radii_grid(n_radii, COARSE_STEP)
    logger.info('  Coarse grid: %d combinations (step=%d)', len(coarse_combos), COARSE_STEP)

    best_max_err = float('inf')
    best_radii_coarse = None

    for idx, radii in enumerate(coarse_combos):
        if idx > 0 and idx % 500 == 0:
            logger.info('    Coarse progress: %d/%d', idx, len(coarse_combos))
        err_pn, err_nfe, _ = eval_fn(radii, cum_tables, r_grid)
        max_err = max(err_pn, err_nfe) if np.isfinite(err_pn) and np.isfinite(err_nfe) else float('inf')
        if max_err < best_max_err:
            best_max_err = max_err
            best_radii_coarse = radii

    if best_radii_coarse is None:
        logger.warning('  Coarse grid found no valid solution')
        return None

    logger.info('  Coarse optimum: radii=%s, max_err=%.4f', best_radii_coarse, best_max_err)

    # Stage 2: fine grid around coarse optimum
    fine_bounds = [
        (max(0, r - FINE_NEIGHBORHOOD), r + FINE_NEIGHBORHOOD)
        for r in best_radii_coarse
    ]
    fine_combos = _generate_radii_grid(n_radii, FINE_STEP, fine_bounds)
    logger.info('  Fine grid: %d combinations (step=%d)', len(fine_combos), FINE_STEP)

    best_model = None
    best_max_err_fine = float('inf')

    for radii in fine_combos:
        err_pn, err_nfe, model = eval_fn(radii, cum_tables, r_grid)
        max_err = max(err_pn, err_nfe) if np.isfinite(err_pn) and np.isfinite(err_nfe) else float('inf')
        if max_err < best_max_err_fine:
            best_max_err_fine = max_err
            best_model = model

    if best_model is not None:
        logger.info('  Fine optimum: radii=%s, err_pN=%.4f, err_NFe=%.4f',
                    best_model['radii'], best_model['err_pN'], best_model['err_NFe'])
    return best_model


def evaluate_baseline(cum_tables, r_grid):
    """Evaluate baseline (current) criterion using existing optimize_radii logic.

    Returns model dict compatible with multi-ring models.
    """
    from sphere_appro.criterion import optimize_radii

    choices = optimize_radii(cum_tables, r_grid)
    mm = choices.get('minimax')
    if mm is None:
        return None

    r1, r2, err_pn, err_nfe = mm
    cri_p, vp = compute_criteria_fast(r1, r2, cum_tables[0], r_grid)
    cri_n, vn = compute_criteria_fast(r1, r2, cum_tables[1], r_grid)
    cri_fe, vfe = compute_criteria_fast(r1, r2, cum_tables[2], r_grid)

    _, border_pn = find_optimal_border(cri_p[vp], cri_n[vn])
    _, border_nfe = find_optimal_border(cri_n[vn], cri_fe[vfe])

    return {
        'radii': [r1, r2],
        'w_pN': np.array([1.0]), 'border_pN': border_pn,
        'w_NFe': np.array([1.0]), 'border_NFe': border_nfe,
        'err_pN': err_pn, 'err_NFe': err_nfe,
    }


def train_variant(variant: str, cum_tables, r_grid):
    """Train one criterion variant. Returns model dict or None.

    Model dict keys: radii, w_pN, border_pN, w_NFe, border_NFe, err_pN, err_NFe
    """
    if variant == 'baseline':
        return evaluate_baseline(cum_tables, r_grid)
    elif variant == '2ring_fisher':
        return optimize_radii_multi(cum_tables, r_grid, n_radii=3, use_minimax_2d=False)
    elif variant == '3ring_fisher':
        return optimize_radii_multi(cum_tables, r_grid, n_radii=4, use_minimax_2d=False)
    elif variant == '2ring_minimax':
        return optimize_radii_multi(cum_tables, r_grid, n_radii=3, use_minimax_2d=True)
    else:
        raise ValueError(f'Unknown variant: {variant}')


# ===================================================================
# Transfer testing
# ===================================================================

def test_variant(variant: str, model: dict, cum_tables, r_grid):
    """Apply a trained model to test data. Returns dict with err_pN, err_NFe."""
    radii = model['radii']

    if variant == 'baseline':
        r1, r2 = radii
        cri_p, vp = compute_criteria_fast(r1, r2, cum_tables[0], r_grid)
        cri_n, vn = compute_criteria_fast(r1, r2, cum_tables[1], r_grid)
        cri_fe, vfe = compute_criteria_fast(r1, r2, cum_tables[2], r_grid)

        border_pn = model['border_pN']
        border_nfe = model['border_NFe']

        err_pn = max(np.mean(cri_p[vp] < border_pn), np.mean(cri_n[vn] > border_pn))
        err_nfe = max(np.mean(cri_n[vn] < border_nfe), np.mean(cri_fe[vfe] > border_nfe))
    else:
        # Multi-ring variants: compute features, project onto trained direction
        feats_by_particle = {}
        for i, particle in enumerate(PARTICLES):
            feats, valid = compute_zone_features(cum_tables[i], r_grid, radii)
            feats_by_particle[particle] = feats[valid]

        w_pn = model['w_pN']
        border_pn = model['border_pN']
        scores_p = feats_by_particle['p'] @ w_pn
        scores_n_pn = feats_by_particle['N'] @ w_pn
        err_pn = max(np.mean(scores_p < border_pn), np.mean(scores_n_pn > border_pn))

        w_nfe = model['w_NFe']
        border_nfe = model['border_NFe']
        scores_n_nfe = feats_by_particle['N'] @ w_nfe
        scores_fe = feats_by_particle['Fe'] @ w_nfe
        err_nfe = max(np.mean(scores_n_nfe < border_nfe), np.mean(scores_fe > border_nfe))

    return {'err_pN': err_pn, 'err_NFe': err_nfe}


def run_all_variants(tables: dict):
    """Run all variants on all datasets, compute transfer matrices.

    Returns:
        summary_rows: list of dicts (per-variant, per-dataset self-test results)
        transfer_rows: list of dicts (full NxN transfer for each variant)
        trained_models: dict of {(variant, atm): model} for plotting
    """
    summary_rows = []
    transfer_rows = []
    trained_models = {}

    # Get r_grid from first available table
    r_grid = next(iter(tables.values()))[0]

    for variant in VARIANT_NAMES:
        logger.info('=== Variant: %s ===', variant)

        for atm_train in ATM_LABELS:
            # Collect cumulative tables for training atmosphere
            cum_list_train = []
            skip = False
            for particle in PARTICLES:
                key = (atm_train, particle)
                if key not in tables:
                    logger.warning('Missing table for %s, skipping', key)
                    skip = True
                    break
                cum_list_train.append(tables[key][1])
            if skip:
                continue

            logger.info('  Training %s on %s ...', variant, atm_train)
            t0 = time.time()
            model = train_variant(variant, cum_list_train, r_grid)
            t1 = time.time()

            if model is None:
                logger.warning('  Training failed for %s on %s', variant, atm_train)
                continue

            logger.info('  Trained in %.1fs: radii=%s, err_pN=%.4f, err_NFe=%.4f',
                        t1 - t0, model['radii'], model['err_pN'], model['err_NFe'])

            trained_models[(variant, atm_train)] = model

            # Self-test result for summary
            radii_str = ','.join(str(r) for r in model['radii'])
            w_pn_str = ','.join(f'{w:.4f}' for w in model['w_pN'])
            w_nfe_str = ','.join(f'{w:.4f}' for w in model['w_NFe'])
            summary_rows.append({
                'variant': variant, 'atm': atm_train,
                'err_pN': model['err_pN'], 'err_NFe': model['err_NFe'],
                'radii': radii_str, 'w_pN': w_pn_str, 'w_NFe': w_nfe_str,
            })

            # Transfer test on all datasets
            for atm_test in ATM_LABELS:
                cum_list_test = []
                skip_test = False
                for particle in PARTICLES:
                    key = (atm_test, particle)
                    if key not in tables:
                        skip_test = True
                        break
                    cum_list_test.append(tables[key][1])
                if skip_test:
                    continue

                if atm_test == atm_train:
                    test_result = {'err_pN': model['err_pN'], 'err_NFe': model['err_NFe']}
                else:
                    test_result = test_variant(variant, model, cum_list_test, r_grid)

                transfer_rows.append({
                    'variant': variant,
                    'train_atm': atm_train, 'test_atm': atm_test,
                    'test_err_pN': test_result['err_pN'],
                    'test_err_NFe': test_result['err_NFe'],
                    'self_err_pN': model['err_pN'],
                    'self_err_NFe': model['err_NFe'],
                })

    return summary_rows, transfer_rows, trained_models


# ===================================================================
# Plotting
# ===================================================================

VARIANT_COLORS = {
    'baseline': '#1f77b4',
    '2ring_fisher': '#ff7f0e',
    '3ring_fisher': '#2ca02c',
    '2ring_minimax': '#d62728',
}


def plot_error_comparison(summary_df: pd.DataFrame) -> None:
    """Grouped bar chart: err_pN and err_NFe by variant, grouped by dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, err_col, title in zip(axes,
                                   ['err_pN', 'err_NFe'],
                                   ['p-N separation error', 'N-Fe separation error']):
        x = np.arange(len(ATM_LABELS))
        n_variants = len(VARIANT_NAMES)
        width = 0.8 / n_variants

        for i, variant in enumerate(VARIANT_NAMES):
            sub = summary_df[summary_df['variant'] == variant]
            vals = [sub[sub['atm'] == atm][err_col].values[0]
                    if len(sub[sub['atm'] == atm]) > 0 else 0
                    for atm in ATM_LABELS]
            offset = (i - n_variants / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=variant,
                   color=VARIANT_COLORS[variant], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(ATM_LABELS)
        ax.set_ylabel('Max classification error')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'error_comparison.png', dpi=150)
    plt.close(fig)
    logger.info('Saved error_comparison.png')


def plot_transfer_heatmaps(transfer_df: pd.DataFrame) -> None:
    """4x4 heatmap of transfer errors for each variant."""
    for variant in VARIANT_NAMES:
        sub = transfer_df[transfer_df['variant'] == variant]
        if len(sub) == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Transfer matrix: {variant}', fontsize=14)

        for ax, err_col, label in zip(axes,
                                       ['test_err_pN', 'test_err_NFe'],
                                       ['p-N', 'N-Fe']):
            pivot = sub.pivot(index='train_atm', columns='test_atm', values=err_col)
            pivot = pivot.reindex(index=ATM_LABELS, columns=ATM_LABELS)
            vals = pivot.values.astype(float)

            im = ax.imshow(vals, cmap='YlOrRd', aspect='auto', vmin=0.3, vmax=0.5)
            ax.set_xticks(range(len(ATM_LABELS)))
            ax.set_xticklabels(ATM_LABELS, fontsize=8)
            ax.set_yticks(range(len(ATM_LABELS)))
            ax.set_yticklabels(ATM_LABELS, fontsize=8)
            ax.set_xlabel('Test')
            ax.set_ylabel('Train')
            for i in range(len(ATM_LABELS)):
                for j in range(len(ATM_LABELS)):
                    v = vals[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=10)
            ax.set_title(label)
            plt.colorbar(im, ax=ax, label='Error')

        plt.tight_layout()
        fname = f'transfer_heatmap_{variant}.png'
        fig.savefig(PLOT_DIR / fname, dpi=150)
        plt.close(fig)
        logger.info('Saved %s', fname)


def plot_degradation_comparison(transfer_df: pd.DataFrame) -> None:
    """Bar chart: mean degradation per variant across all cross-pairs."""
    rows = []
    for variant in VARIANT_NAMES:
        sub = transfer_df[transfer_df['variant'] == variant]
        cross = sub[sub['train_atm'] != sub['test_atm']]
        if len(cross) == 0:
            continue
        deg_pn = (cross['test_err_pN'] - cross['self_err_pN']).mean()
        deg_nfe = (cross['test_err_NFe'] - cross['self_err_NFe']).mean()
        rows.append({'variant': variant, 'deg_pN': deg_pn, 'deg_NFe': deg_nfe})

    if not rows:
        return

    deg_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(deg_df))
    width = 0.35
    ax.bar(x - width / 2, deg_df['deg_pN'], width, label='p-N degradation',
           color='#1f77b4', alpha=0.8)
    ax.bar(x + width / 2, deg_df['deg_NFe'], width, label='N-Fe degradation',
           color='#ff7f0e', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(deg_df['variant'], fontsize=9)
    ax.set_ylabel('Mean degradation (cross-atm - self)')
    ax.set_title('Transfer degradation by variant')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'degradation_comparison.png', dpi=150)
    plt.close(fig)
    logger.info('Saved degradation_comparison.png')


def plot_scatter_2d(tables: dict, trained_models: dict) -> None:
    """2D scatter (f1, f2) for 3 particles + Fisher boundary, one per dataset."""
    particle_colors = {'p': '#d62728', 'N': '#2ca02c', 'Fe': '#1f77b4'}

    for atm in ATM_LABELS:
        model_key = ('2ring_fisher', atm)
        if model_key not in trained_models:
            continue
        model = trained_models[model_key]
        radii = model['radii']
        r_grid = next(iter(tables.values()))[0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'2-ring features: {atm} (radii={radii})', fontsize=13)

        # Gather features for all particles
        feats_all = {}
        for particle in PARTICLES:
            key = (atm, particle)
            if key not in tables:
                continue
            feats, valid = compute_zone_features(tables[key][1], r_grid, radii)
            feats_all[particle] = feats[valid]

        # Left panel: raw f1 vs f2
        ax = axes[0]
        for particle in PARTICLES:
            if particle not in feats_all:
                continue
            f = feats_all[particle]
            ax.scatter(f[:, 0], f[:, 1], s=1, alpha=0.1, c=particle_colors[particle],
                      label=particle, rasterized=True)
        ax.set_xlabel('f1 = log(Z1/Z2)')
        ax.set_ylabel('f2 = log(Z2/Z3)')
        ax.legend(markerscale=10)
        ax.set_title('Feature space')
        ax.grid(True, alpha=0.3)

        # Right panel: Fisher scores histogram (p-N direction)
        ax = axes[1]
        w_pn = model['w_pN']
        for particle in PARTICLES:
            if particle not in feats_all:
                continue
            scores = feats_all[particle] @ w_pn
            ax.hist(scores, bins=100, alpha=0.5, color=particle_colors[particle],
                    label=particle, density=True)
        ax.axvline(model['border_pN'], color='black', linestyle='--',
                   label=f'border_pN={model["border_pN"]:.3f}')
        ax.set_xlabel('Fisher score (p-N direction)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)
        ax.set_title('Fisher p-N score distribution')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'scatter_2d_{atm}.png'
        fig.savefig(PLOT_DIR / fname, dpi=150)
        plt.close(fig)
        logger.info('Saved %s', fname)


def plot_fisher_weights(trained_models: dict) -> None:
    """Bar chart of Fisher weights per variant per dataset."""
    fisher_variants = ['2ring_fisher', '3ring_fisher']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fisher LDA weights by variant and dataset', fontsize=13)

    for ax, err_label, w_key in zip(axes, ['p-N', 'N-Fe'], ['w_pN', 'w_NFe']):
        all_bars = []
        labels = []
        for variant in fisher_variants:
            for atm in ATM_LABELS:
                key = (variant, atm)
                if key not in trained_models:
                    continue
                model = trained_models[key]
                w = model[w_key]
                all_bars.append(w)
                labels.append(f'{variant}\n{atm}')

        if not all_bars:
            continue

        max_dim = max(len(w) for w in all_bars)
        x = np.arange(len(all_bars))
        width = 0.8 / max_dim
        dim_labels = [f'w{i+1}' for i in range(max_dim)]

        for d in range(max_dim):
            vals = [w[d] if d < len(w) else 0 for w in all_bars]
            offset = (d - max_dim / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=dim_labels[d], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('Fisher weight')
        ax.set_title(f'Fisher weights: {err_label}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'fisher_weights.png', dpi=150)
    plt.close(fig)
    logger.info('Saved fisher_weights.png')


# ===================================================================
# Report
# ===================================================================

class Report:
    def __init__(self):
        self.sections: list[str] = []

    def add(self, title: str, body: str) -> None:
        self.sections.append(f'## {title}\n\n{body}\n')

    def save(self, path: Path) -> None:
        header = '# Multi-Ring Criterion Study Report\n\n'
        content = header + '\n'.join(self.sections)
        path.write_text(content)
        logger.info('Report saved to %s', path)


def generate_report(summary_df, transfer_df):
    """Generate markdown report."""
    report = Report()

    # Data summary
    report.add('Configuration', (
        f'Variants tested: {", ".join(VARIANT_NAMES)}\n\n'
        f'Datasets: {", ".join(ATM_LABELS)}\n\n'
        f'Radial step: {RADIAL_STEP} mm, R_max: {R_MAX} mm\n\n'
        f'Coarse grid step: {COARSE_STEP} mm, Fine grid step: {FINE_STEP} mm\n'
    ))

    # Self-test errors comparison
    report.add('Self-Test Errors', summary_df.to_markdown(index=False))

    # Transfer matrix per variant
    for variant in VARIANT_NAMES:
        sub = transfer_df[transfer_df['variant'] == variant]
        if len(sub) == 0:
            continue
        report.add(f'Transfer Matrix: {variant}', sub.to_markdown(index=False))

    # Degradation summary
    deg_rows = []
    for variant in VARIANT_NAMES:
        sub = transfer_df[transfer_df['variant'] == variant]
        cross = sub[sub['train_atm'] != sub['test_atm']]
        if len(cross) == 0:
            continue
        deg_pn = (cross['test_err_pN'] - cross['self_err_pN']).mean()
        deg_nfe = (cross['test_err_NFe'] - cross['self_err_NFe']).mean()
        max_deg_pn = (cross['test_err_pN'] - cross['self_err_pN']).max()
        max_deg_nfe = (cross['test_err_NFe'] - cross['self_err_NFe']).max()
        deg_rows.append({
            'variant': variant,
            'mean_deg_pN': f'{deg_pn:.4f}', 'max_deg_pN': f'{max_deg_pn:.4f}',
            'mean_deg_NFe': f'{deg_nfe:.4f}', 'max_deg_NFe': f'{max_deg_nfe:.4f}',
        })
    if deg_rows:
        deg_df = pd.DataFrame(deg_rows)
        report.add('Transfer Degradation Summary', deg_df.to_markdown(index=False))

    return report


# ===================================================================
# Main
# ===================================================================

if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info('=== Loading data ===')
    df = load_data()

    # Build cumulative tables (shared across all variants)
    logger.info('=== Building cumulative tables ===')
    t0 = time.time()
    tables = build_cum_tables(df)
    t1 = time.time()
    logger.info('Cumulative tables built in %.1f s', t1 - t0)

    # Run all variants
    logger.info('=== Running all variants ===')
    t0 = time.time()
    summary_rows, transfer_rows, trained_models = run_all_variants(tables)
    t1 = time.time()
    logger.info('All variants completed in %.1f s', t1 - t0)

    # Save results
    summary_df = pd.DataFrame(summary_rows)
    transfer_df = pd.DataFrame(transfer_rows)

    summary_df.to_parquet(OUTPUT_DIR / 'comparison_summary.parquet', index=False)
    logger.info('Saved comparison_summary.parquet')
    transfer_df.to_parquet(OUTPUT_DIR / 'transfer_matrices.parquet', index=False)
    logger.info('Saved transfer_matrices.parquet')

    # Plots
    logger.info('=== Generating plots ===')
    plot_error_comparison(summary_df)
    plot_transfer_heatmaps(transfer_df)
    plot_degradation_comparison(transfer_df)
    plot_scatter_2d(tables, trained_models)
    plot_fisher_weights(trained_models)

    # Report
    logger.info('=== Generating report ===')
    report = generate_report(summary_df, transfer_df)
    report.save(OUTPUT_DIR / 'report.md')

    logger.info('=== Done ===')
