#!/usr/bin/env python3
"""Compare Sb detector (atm00) vs standard detector (atm01) for SPHERE-3."""
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from sphere_appro.criterion import (
    precompute_cumulative_tables, compute_criteria_fast,
    find_optimal_border, optimize_radii,
)
from sphere_appro.config import R2_MAX, RADIAL_STEP

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ATM_FILES = {
    'atm00_sb': Path('results_sb.parquet'),
    'atm01': Path('results_fnew.parquet'),
}
ATM_LABELS = list(ATM_FILES.keys())
ATM_COLORS = {'atm00_sb': '#d62728', 'atm01': '#1f77b4'}
OUTPUT_DIR = Path('analysis_output_sb_comparison')
PLOT_DIR = OUTPUT_DIR / 'plots'
SHAPE_PARAMS = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'R_ch', 'sw']
KEY_PARAMS = ['p0', 'p1', 'p4', 'p5', 'R_ch', 'sw']
QUALITY_METRICS = ['fval', 'chi2_ndf', 'max_abs_d', 'mean_abs_d']
PARTICLES = ['p', 'N', 'Fe']
ATM_PAIRS = [('atm00_sb', 'atm01')]


# ===================================================================
# Task 0 — Data loading
# ===================================================================

def load_data() -> pd.DataFrame:
    """Read the two parquet files, tag with 'atm' column, concatenate."""
    frames = []
    for label, path in ATM_FILES.items():
        logger.info('Loading %s from %s ...', label, path)
        df = pd.read_parquet(path)
        df['atm'] = label
        frames.append(df)
        logger.info('  %s: %d rows, columns: %s', label, len(df), list(df.columns))
    combined = pd.concat(frames, ignore_index=True)
    logger.info('Combined dataset: %d rows', len(combined))
    return combined


# ===================================================================
# Task 1 — LDF parameter comparison
# ===================================================================

def compute_param_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-atmosphere, per-particle descriptive statistics for shape params."""
    rows = []
    for atm in ATM_LABELS:
        for particle in PARTICLES:
            sub = df[(df['atm'] == atm) & (df['particle'] == particle)]
            for param in SHAPE_PARAMS:
                vals = sub[param].dropna()
                rows.append({
                    'atm': atm, 'particle': particle, 'param': param,
                    'count': len(vals),
                    'mean': vals.mean(),
                    'std': vals.std(),
                    'median': vals.median(),
                    'q25': vals.quantile(0.25),
                    'q75': vals.quantile(0.75),
                })
    return pd.DataFrame(rows)


def compute_relative_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Relative difference of medians for the atmosphere pair."""
    rows = []
    for particle in PARTICLES:
        for param in SHAPE_PARAMS:
            medians = {}
            for atm in ATM_LABELS:
                sub = df[(df['atm'] == atm) & (df['particle'] == particle)]
                medians[atm] = sub[param].median()
            for a1, a2 in ATM_PAIRS:
                ref = medians[a1]
                if ref == 0:
                    rd = np.nan
                else:
                    rd = (medians[a2] - medians[a1]) / abs(ref)
                rows.append({
                    'particle': particle, 'param': param,
                    'pair': f'{a1}_vs_{a2}',
                    'median_1': medians[a1], 'median_2': medians[a2],
                    'rel_diff': rd,
                })
    return pd.DataFrame(rows)


def compute_ks_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Two-sample KS test for every (particle, param, atm pair)."""
    rows = []
    for particle in PARTICLES:
        for param in SHAPE_PARAMS:
            samples = {}
            for atm in ATM_LABELS:
                sub = df[(df['atm'] == atm) & (df['particle'] == particle)]
                samples[atm] = sub[param].dropna().values
            for a1, a2 in ATM_PAIRS:
                if len(samples[a1]) < 2 or len(samples[a2]) < 2:
                    continue
                ks_stat, p_val = stats.ks_2samp(samples[a1], samples[a2])
                rows.append({
                    'particle': particle, 'param': param,
                    'pair': f'{a1}_vs_{a2}',
                    'ks_stat': ks_stat, 'p_value': p_val,
                })
    return pd.DataFrame(rows)


def plot_param_boxplots(df: pd.DataFrame) -> None:
    """Box-plots of key LDF params split by atmosphere, one figure per particle."""
    for particle in PARTICLES:
        sub = df[df['particle'] == particle]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'LDF parameter distributions — {particle}', fontsize=14)
        for ax, param in zip(axes.ravel(), KEY_PARAMS):
            data_by_atm = [sub[sub['atm'] == a][param].dropna().values for a in ATM_LABELS]
            bp = ax.boxplot(data_by_atm, tick_labels=ATM_LABELS, patch_artist=True)
            colors = [ATM_COLORS[a] for a in ATM_LABELS]
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            ax.set_title(param)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(PLOT_DIR / f'param_boxplots_{particle}.png', dpi=150)
        plt.close(fig)
        logger.info('Saved param_boxplots_%s.png', particle)


def plot_reldiff_heatmaps(reldiff: pd.DataFrame) -> None:
    """Heatmap of relative median differences for the atmosphere pair."""
    for pair_label in reldiff['pair'].unique():
        sub = reldiff[reldiff['pair'] == pair_label]
        pivot = sub.pivot(index='param', columns='particle', values='rel_diff')
        pivot = pivot.reindex(index=SHAPE_PARAMS, columns=PARTICLES)
        fig, ax = plt.subplots(figsize=(6, 8))
        im = ax.imshow(pivot.values.astype(float), cmap='RdBu_r', aspect='auto',
                        vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(PARTICLES)))
        ax.set_xticklabels(PARTICLES)
        ax.set_yticks(range(len(SHAPE_PARAMS)))
        ax.set_yticklabels(SHAPE_PARAMS)
        for i in range(len(SHAPE_PARAMS)):
            for j in range(len(PARTICLES)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9)
        plt.colorbar(im, ax=ax, label='Relative difference of medians')
        ax.set_title(f'Relative median diff: {pair_label}')
        plt.tight_layout()
        fname = f'reldiff_heatmap_{pair_label}.png'
        fig.savefig(PLOT_DIR / fname, dpi=150)
        plt.close(fig)
        logger.info('Saved %s', fname)


# ===================================================================
# Task 2 — Criterion transfer
# ===================================================================

def build_cum_tables(df: pd.DataFrame) -> dict:
    """Build cumulative tables keyed by (atm, particle)."""
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
                r_max=R2_MAX, dr=RADIAL_STEP,
            )
            tables[(atm, particle)] = (r_grid, cum)
            logger.info('  cum table (%s, %s): %d events, grid %d',
                        atm, particle, cum.shape[0], cum.shape[1])
    return tables


def train_criterion(tables: dict, atm: str):
    """Train criterion on a single atmosphere: optimize_radii, find_optimal_border."""
    cum_list = []
    r_grid = None
    for particle in PARTICLES:
        key = (atm, particle)
        if key not in tables:
            return None
        rg, cum = tables[key]
        cum_list.append(cum)
        if r_grid is None:
            r_grid = rg

    choices = optimize_radii(cum_list, r_grid)
    mm = choices.get('minimax')
    if mm is None:
        return None

    r1, r2, err_pn, err_nfe = mm
    cri_p, vp = compute_criteria_fast(r1, r2, cum_list[0], r_grid)
    cri_n, vn = compute_criteria_fast(r1, r2, cum_list[1], r_grid)
    cri_fe, vfe = compute_criteria_fast(r1, r2, cum_list[2], r_grid)

    _, border_pn = find_optimal_border(cri_p[vp], cri_n[vn])
    _, border_nfe = find_optimal_border(cri_n[vn], cri_fe[vfe])

    return {
        'r1': r1, 'r2': r2,
        'border_pN': border_pn, 'border_NFe': border_nfe,
        'err_pN': err_pn, 'err_NFe': err_nfe,
    }


def test_criterion(tables: dict, atm_test: str, trained: dict):
    """Apply trained criterion on a different atmosphere."""
    r1, r2 = trained['r1'], trained['r2']
    border_pn, border_nfe = trained['border_pN'], trained['border_NFe']

    cum_list = []
    r_grid = None
    for particle in PARTICLES:
        key = (atm_test, particle)
        if key not in tables:
            return None
        rg, cum = tables[key]
        cum_list.append(cum)
        if r_grid is None:
            r_grid = rg

    cri_p, vp = compute_criteria_fast(r1, r2, cum_list[0], r_grid)
    cri_n, vn = compute_criteria_fast(r1, r2, cum_list[1], r_grid)
    cri_fe, vfe = compute_criteria_fast(r1, r2, cum_list[2], r_grid)

    err_pn_p = np.mean(cri_p[vp] < border_pn)
    err_pn_n = np.mean(cri_n[vn] > border_pn)
    err_nfe_n = np.mean(cri_n[vn] < border_nfe)
    err_nfe_fe = np.mean(cri_fe[vfe] > border_nfe)

    err_pn = max(err_pn_p, err_pn_n)
    err_nfe = max(err_nfe_n, err_nfe_fe)

    return {
        'err_pN': err_pn, 'err_NFe': err_nfe,
        'err_pN_detail': (err_pn_p, err_pn_n),
        'err_NFe_detail': (err_nfe_n, err_nfe_fe),
    }


def compute_transfer_matrix(tables: dict) -> pd.DataFrame:
    """Build full transfer matrix: train on atm_i, test on atm_j."""
    rows = []
    for atm_train in ATM_LABELS:
        logger.info('Training criterion on %s ...', atm_train)
        trained = train_criterion(tables, atm_train)
        if trained is None:
            logger.warning('Training failed for %s', atm_train)
            continue
        for atm_test in ATM_LABELS:
            logger.info('  Testing on %s ...', atm_test)
            result = test_criterion(tables, atm_test, trained)
            if result is None:
                continue
            rows.append({
                'train_atm': atm_train, 'test_atm': atm_test,
                'r1': trained['r1'], 'r2': trained['r2'],
                'border_pN': trained['border_pN'],
                'border_NFe': trained['border_NFe'],
                'train_err_pN': trained['err_pN'],
                'train_err_NFe': trained['err_NFe'],
                'test_err_pN': result['err_pN'],
                'test_err_NFe': result['err_NFe'],
            })
    return pd.DataFrame(rows)


def plot_transfer_heatmaps(transfer_df: pd.DataFrame) -> None:
    """Heatmap of test errors for p-N and N-Fe."""
    for err_col, title_suffix in [('test_err_pN', 'p-N'), ('test_err_NFe', 'N-Fe')]:
        pivot = transfer_df.pivot(index='train_atm', columns='test_atm', values=err_col)
        pivot = pivot.reindex(index=ATM_LABELS, columns=ATM_LABELS)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(pivot.values.astype(float), cmap='YlOrRd', aspect='auto',
                        vmin=0, vmax=0.5)
        ax.set_xticks(range(len(ATM_LABELS)))
        ax.set_xticklabels(ATM_LABELS)
        ax.set_yticks(range(len(ATM_LABELS)))
        ax.set_yticklabels(ATM_LABELS)
        ax.set_xlabel('Test atmosphere')
        ax.set_ylabel('Train atmosphere')
        for i in range(len(ATM_LABELS)):
            for j in range(len(ATM_LABELS)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=12)
        plt.colorbar(im, ax=ax, label='Max classification error')
        ax.set_title(f'Criterion transfer: {title_suffix}')
        plt.tight_layout()
        fname = f'transfer_heatmap_{err_col}.png'
        fig.savefig(PLOT_DIR / fname, dpi=150)
        plt.close(fig)
        logger.info('Saved %s', fname)


def plot_transfer_degradation(transfer_df: pd.DataFrame) -> None:
    """Bar chart showing error degradation when transferring criterion."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, err_col, title in zip(axes,
                                   ['test_err_pN', 'test_err_NFe'],
                                   ['p-N separation', 'N-Fe separation']):
        diag = transfer_df[transfer_df['train_atm'] == transfer_df['test_atm']].copy()
        off_diag = transfer_df[transfer_df['train_atm'] != transfer_df['test_atm']].copy()

        diag_vals = diag.set_index('train_atm')[err_col]
        off_vals = off_diag.set_index('train_atm')[err_col]

        x = np.arange(len(ATM_LABELS))
        width = 0.35
        ax.bar(x - width / 2, [diag_vals.get(a, 0) for a in ATM_LABELS],
               width, label='Self-test', color='#1f77b4', alpha=0.8)
        ax.bar(x + width / 2, [off_vals.get(a, 0) for a in ATM_LABELS],
               width, label='Cross-atm', color='#ff7f0e', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ATM_LABELS)
        ax.set_ylabel('Max classification error')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'transfer_degradation.png', dpi=150)
    plt.close(fig)
    logger.info('Saved transfer_degradation.png')


# ===================================================================
# Task 3 — Approximation quality
# ===================================================================

def compute_quality_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-atmosphere, per-particle descriptive statistics for quality metrics."""
    rows = []
    for atm in ATM_LABELS:
        for particle in PARTICLES:
            sub = df[(df['atm'] == atm) & (df['particle'] == particle)]
            for metric in QUALITY_METRICS:
                vals = sub[metric].dropna()
                if len(vals) == 0:
                    continue
                rows.append({
                    'atm': atm, 'particle': particle, 'metric': metric,
                    'count': len(vals),
                    'mean': vals.mean(),
                    'std': vals.std(),
                    'median': vals.median(),
                    'q25': vals.quantile(0.25),
                    'q75': vals.quantile(0.75),
                    'q95': vals.quantile(0.95),
                })
    return pd.DataFrame(rows)


def plot_quality_violin(df: pd.DataFrame) -> None:
    """Violin plots of quality metrics, split by atmosphere and particle."""
    for metric in QUALITY_METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        fig.suptitle(f'Approximation quality: {metric}', fontsize=14)
        for ax, particle in zip(axes, PARTICLES):
            data_by_atm = []
            for atm in ATM_LABELS:
                vals = df[(df['atm'] == atm) & (df['particle'] == particle)][metric].dropna().values
                data_by_atm.append(vals)
            parts = ax.violinplot(data_by_atm, showmeans=True, showmedians=True)
            colors = [ATM_COLORS[a] for a in ATM_LABELS]
            for pc, c in zip(parts['bodies'], colors):
                pc.set_facecolor(c)
                pc.set_alpha(0.6)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(ATM_LABELS)
            ax.set_title(particle)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel(metric)
        plt.tight_layout()
        fname = f'quality_violin_{metric}.png'
        fig.savefig(PLOT_DIR / fname, dpi=150)
        plt.close(fig)
        logger.info('Saved %s', fname)


def plot_quality_cdf(df: pd.DataFrame) -> None:
    """CDF plots of quality metrics for each particle, overlaid by atmosphere."""
    for metric in QUALITY_METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'CDF of {metric}', fontsize=14)
        for ax, particle in zip(axes, PARTICLES):
            for atm in ATM_LABELS:
                vals = df[(df['atm'] == atm) & (df['particle'] == particle)][metric].dropna().values
                if len(vals) == 0:
                    continue
                sorted_vals = np.sort(vals)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                ax.plot(sorted_vals, cdf, label=atm, color=ATM_COLORS[atm], linewidth=1.5)
            ax.set_title(particle)
            ax.set_xlabel(metric)
            ax.set_ylabel('CDF')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f'quality_cdf_{metric}.png'
        fig.savefig(PLOT_DIR / fname, dpi=150)
        plt.close(fig)
        logger.info('Saved %s', fname)


def plot_quality_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of median quality metrics per (atm, particle, angle)."""
    for metric in QUALITY_METRICS:
        if 'angle' not in df.columns:
            logger.warning('No angle column, skipping quality heatmap for %s', metric)
            return
        angles = sorted(df['angle'].dropna().unique())
        fig, axes = plt.subplots(1, len(PARTICLES), figsize=(6 * len(PARTICLES), max(4, len(angles) * 0.4 + 2)))
        fig.suptitle(f'Median {metric} by angle', fontsize=14)
        if len(PARTICLES) == 1:
            axes = [axes]
        for ax, particle in zip(axes, PARTICLES):
            sub = df[df['particle'] == particle]
            pivot = sub.groupby(['angle', 'atm'])[metric].median().unstack('atm')
            pivot = pivot.reindex(index=angles, columns=ATM_LABELS)
            im = ax.imshow(pivot.values.astype(float), cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(ATM_LABELS)))
            ax.set_xticklabels(ATM_LABELS)
            ax.set_yticks(range(len(angles)))
            ax.set_yticklabels([f'{a:.0f}' if isinstance(a, (int, float)) else str(a) for a in angles])
            ax.set_ylabel('Angle')
            ax.set_title(particle)
            plt.colorbar(im, ax=ax, label=f'median {metric}')
        plt.tight_layout()
        fname = f'quality_heatmap_{metric}.png'
        fig.savefig(PLOT_DIR / fname, dpi=150)
        plt.close(fig)
        logger.info('Saved %s', fname)


# ===================================================================
# Task 4 — Report
# ===================================================================

class Report:
    """Collects markdown sections and writes report.md."""

    def __init__(self):
        self.sections: list[str] = []

    def add(self, title: str, body: str) -> None:
        self.sections.append(f'## {title}\n\n{body}\n')

    def save(self, path: Path) -> None:
        header = '# Sb (atm00) vs atm01 Comparison Report\n\n'
        content = header + '\n'.join(self.sections)
        path.write_text(content)
        logger.info('Report saved to %s', path)


# ===================================================================
# main
# ===================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    report = Report()

    # ---------------------------------------------------------------
    # Task 0: Load data
    # ---------------------------------------------------------------
    logger.info('=== Task 0: Loading data ===')
    df = load_data()
    report.add('Data Summary', (
        f'Total events: {len(df)}\n\n'
        + '\n'.join(
            f'- **{atm}**: {len(df[df["atm"] == atm])} events'
            for atm in ATM_LABELS
        )
    ))

    # ---------------------------------------------------------------
    # Task 1: LDF parameter comparison
    # ---------------------------------------------------------------
    logger.info('=== Task 1: LDF parameter comparison ===')
    param_stats = compute_param_stats(df)
    reldiff = compute_relative_diff(df)
    ks_results = compute_ks_tests(df)

    plot_param_boxplots(df)
    plot_reldiff_heatmaps(reldiff)

    sig_ks = ks_results[ks_results['p_value'] < 0.01]
    report.add('LDF Parameter Comparison', (
        f'Computed stats for {len(SHAPE_PARAMS)} parameters x {len(PARTICLES)} particles x '
        f'{len(ATM_LABELS)} atmospheres.\n\n'
        f'KS tests with p < 0.01: {len(sig_ks)} / {len(ks_results)}\n\n'
        '### Largest relative differences (|rel_diff| > 0.1)\n\n'
        + reldiff[reldiff['rel_diff'].abs() > 0.1].to_markdown(index=False)
    ))

    summary = pd.concat([
        param_stats.assign(table='param_stats'),
        reldiff.assign(table='reldiff'),
        ks_results.assign(table='ks_test'),
    ], ignore_index=True)
    summary.to_parquet(OUTPUT_DIR / 'summary.parquet', index=False)
    logger.info('Saved summary.parquet')

    # ---------------------------------------------------------------
    # Task 2: Criterion transfer
    # ---------------------------------------------------------------
    logger.info('=== Task 2: Criterion transfer ===')
    t0 = time.time()

    logger.info('Building cumulative tables ...')
    t_tables_start = time.time()
    tables = build_cum_tables(df)
    t_tables_end = time.time()
    logger.info('Cumulative tables built in %.1f s', t_tables_end - t_tables_start)

    logger.info('Computing overall transfer matrix ...')
    t_transfer_start = time.time()
    transfer_df = compute_transfer_matrix(tables)
    t_transfer_end = time.time()
    logger.info('Overall transfer matrix computed in %.1f s', t_transfer_end - t_transfer_start)

    transfer_df.to_parquet(OUTPUT_DIR / 'criterion_transfer.parquet', index=False)
    logger.info('Saved criterion_transfer.parquet')

    plot_transfer_heatmaps(transfer_df)
    plot_transfer_degradation(transfer_df)

    # Per-slice transfers (energy, height)
    logger.info('Computing per-slice criterion transfers ...')
    t_slices_start = time.time()
    slice_rows = []
    slices = df.groupby(['energy', 'height']).size().reset_index(name='count')
    for _, row in slices.iterrows():
        energy, height = row['energy'], row['height']
        slice_df = df[(df['energy'] == energy) & (df['height'] == height)]
        logger.info('  Slice energy=%s height=%s: %d events', energy, height, len(slice_df))

        slice_tables = {}
        skip_slice = False
        for atm in ATM_LABELS:
            for particle in PARTICLES:
                sub = slice_df[(slice_df['atm'] == atm) & (slice_df['particle'] == particle)]
                mask = sub[SHAPE_PARAMS].isna().any(axis=1) | np.isinf(sub[SHAPE_PARAMS].values).any(axis=1)
                sub = sub[~mask]
                if len(sub) < 10:
                    skip_slice = True
                    break
                r_grid, cum = precompute_cumulative_tables(
                    sub['p0'].values, sub['p1'].values,
                    sub['p2'].values, sub['p3'].values,
                    sub['p4'].values, sub['p5'].values,
                    sub['p6'].values, sub['R_ch'].values,
                    sub['sw'].values,
                    r_max=R2_MAX, dr=RADIAL_STEP,
                )
                slice_tables[(atm, particle)] = (r_grid, cum)
            if skip_slice:
                break
        if skip_slice:
            logger.warning('  Skipping slice energy=%s height=%s (insufficient data)', energy, height)
            continue

        for atm_train in ATM_LABELS:
            trained = train_criterion(slice_tables, atm_train)
            if trained is None:
                continue
            for atm_test in ATM_LABELS:
                result = test_criterion(slice_tables, atm_test, trained)
                if result is None:
                    continue
                slice_rows.append({
                    'energy': energy, 'height': height,
                    'train_atm': atm_train, 'test_atm': atm_test,
                    'r1': trained['r1'], 'r2': trained['r2'],
                    'border_pN': trained['border_pN'],
                    'border_NFe': trained['border_NFe'],
                    'train_err_pN': trained['err_pN'],
                    'train_err_NFe': trained['err_NFe'],
                    'test_err_pN': result['err_pN'],
                    'test_err_NFe': result['err_NFe'],
                })

    t_slices_end = time.time()
    logger.info('Per-slice transfers computed in %.1f s', t_slices_end - t_slices_start)

    if slice_rows:
        slice_transfer_df = pd.DataFrame(slice_rows)
        slice_transfer_df.to_parquet(OUTPUT_DIR / 'criterion_transfer_slices.parquet', index=False)
        logger.info('Saved criterion_transfer_slices.parquet (%d rows)', len(slice_transfer_df))
    else:
        slice_transfer_df = pd.DataFrame()
        logger.warning('No per-slice transfer results')

    t1 = time.time()
    logger.info('=== Criterion transfer block total: %.1f s ===', t1 - t0)

    transfer_body = 'Overall transfer matrix:\n\n'
    if not transfer_df.empty:
        transfer_body += transfer_df.to_markdown(index=False) + '\n\n'
    else:
        transfer_body += '(no results)\n\n'
    if not slice_transfer_df.empty:
        transfer_body += f'Per-slice transfers: {len(slice_transfer_df)} rows computed.\n'
    report.add('Criterion Transfer', transfer_body)

    # ---------------------------------------------------------------
    # Task 3: Approximation quality
    # ---------------------------------------------------------------
    logger.info('=== Task 3: Approximation quality ===')
    quality_stats = compute_quality_stats(df)

    plot_quality_violin(df)
    plot_quality_cdf(df)
    plot_quality_heatmap(df)

    quality_body = 'Quality metric medians per atmosphere:\n\n'
    quality_pivot = quality_stats.pivot_table(
        index=['metric', 'particle'], columns='atm', values='median',
    )
    quality_body += quality_pivot.to_markdown() + '\n'
    report.add('Approximation Quality', quality_body)

    # ---------------------------------------------------------------
    # Task 4: Save report
    # ---------------------------------------------------------------
    logger.info('=== Task 4: Saving report ===')
    report.save(OUTPUT_DIR / 'report.md')
    logger.info('All done. Output in %s', OUTPUT_DIR)


if __name__ == '__main__':
    main()
