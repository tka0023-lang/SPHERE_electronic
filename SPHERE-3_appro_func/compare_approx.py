#!/usr/bin/env python3
"""
Comparison tool for moshits and phels LDF approximations.

This module provides functionality to:
1. Load and process both moshits and phels event files
2. Compute LDF (Lateral Distribution Function) for each format
3. Match events between moshits and phels by filename pattern
4. Compare approximation parameters and quality metrics
5. Generate comparison visualizations
"""

import os
import re
import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from functools import partial

import numpy as np
import pandas as pd

# Import from sphere_appro package
from sphere_appro import ldf_model as func
from sphere_appro.optimizer import fit_ldf as minimize_with_restarts, FitResult
from sphere_appro.io_data import (
    load_event_header as get_first_row,
    load_event_file as load_data,
    load_detector_geometry as load_detector_coordinates,
    build_ldf,
    compute_peak_center,
    DetectorGeometry,
    LDF,
)
from sphere_appro.config import (
    GEOMETRY_TYPE,
    N_TOP_PEAKS,
    MAX_CENTER_DISTANCE,
    MAX_PEAK_DISTANCE,
    MIN_TOTAL_INTENSITY_DEFAULT as MIN_TOTAL_INTENSITY,
    PIXEL_SKIP,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MOSHITS_ROOT = Path('/Users/vladimirivanov/Projects/SPHERE/test/moshits')
DEFAULT_PHELS_ROOT = Path('/Users/vladimirivanov/Projects/SPHERE/test/phels')
DEFAULT_PIXEL_DATA_PATH = Path('SPHERE3_pixel_data_A.dat')
DEFAULT_OUTPUT_DIR = Path('comparison_results')

# Conversion factor for phels coordinates
# Note: After checking real data, xx/yy are already in mm (values like -191.897)
# No conversion needed
PHELS_COORD_SCALE = 1.0  # Already in mm

# ============================================================================
# PHELS DATA LOADING
# ============================================================================

def load_phels_header(file_path: Path) -> Dict[str, Any]:
    """
    Parse metadata from the first row of phels file.

    Format identical to moshits: clone_num h x_center y_center event_num

    Args:
        file_path: Path to phels file

    Returns:
        Dictionary with metadata
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Phels file not found: {file_path}")

    with open(file_path, 'r') as f:
        first_row = f.readline().rstrip()

    values = first_row.split()
    if len(values) < 5:
        raise ValueError(f"Expected at least 5 values in header, got {len(values)}")

    return {
        'clone_num': int(values[0]),
        'h': float(values[1]),
        'x_center': float(values[2]),
        'y_center': float(values[3]),
        'event_num': int(values[4])
    }


def load_phels_data(file_path: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Load phels file data.

    Args:
        file_path: Path to phels file

    Returns:
        Tuple (header_dict, data_dataframe)
        Data columns: ii, jj, kk, mmm, xx, yy, tt
        xx, yy are converted to mm (multiplied by 1000)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Phels file not found: {file_path}")

    header = load_phels_header(file_path)

    data = pd.read_csv(
        file_path,
        header=None,
        sep=r'\s+',
        skiprows=1,
        names=['ii', 'jj', 'kk', 'mmm', 'xx', 'yy', 'tt']
    )

    # Convert coordinates from meters to mm
    data['xx'] = data['xx'] * PHELS_COORD_SCALE
    data['yy'] = data['yy'] * PHELS_COORD_SCALE

    logger.debug(f"Loaded {len(data)} phels hits from {file_path}")

    return header, data


# ============================================================================
# LDF COMPUTATION FOR PHELS
# ============================================================================

def compute_ldf_phels(data: pd.DataFrame) -> Tuple[pd.DataFrame, float, float, float, float]:
    """
    Compute LDF for phels data.

    Groups hits by (ii, jj) cell and computes:
    - I: count of hits per cell
    - x, y: mean coordinates per cell (in mm)

    Args:
        data: DataFrame with phels hits (xx, yy already in mm)

    Returns:
        Tuple (ldf, x_center, y_center, x_peak, y_peak)
    """
    if data.empty:
        logger.warning("Empty phels data")
        return pd.DataFrame({'I': [], 'x': [], 'y': []}), 0.0, 0.0, 0.0, 0.0

    # Group by (ii, jj) and aggregate
    grouped = data.groupby(['ii', 'jj']).agg({
        'xx': 'mean',
        'yy': 'mean',
        'kk': 'count'  # count hits per cell
    }).rename(columns={'kk': 'I', 'xx': 'x', 'yy': 'y'})

    ldf = grouped.reset_index(drop=True)
    ldf['I'] = ldf['I'].astype(float)

    total_I = ldf['I'].sum()
    if total_I == 0:
        return ldf, 0.0, 0.0, 0.0, 0.0

    # Compute weighted center (same formula as main.py)
    x_center = np.sqrt((ldf['x']**2 * ldf['I']).sum() / total_I)
    y_center = np.sqrt((ldf['y']**2 * ldf['I']).sum() / total_I)

    # Determine sign from max intensity position
    max_I_idx = ldf['I'].idxmax()
    if ldf.at[max_I_idx, 'x'] < 0:
        x_center = -x_center
    if ldf.at[max_I_idx, 'y'] < 0:
        y_center = -y_center

    # Peak coordinates from top N cells
    topN = ldf.nlargest(N_TOP_PEAKS, 'I')
    x_peak = topN['x'].mean()
    y_peak = topN['y'].mean()

    return ldf, x_center, y_center, x_peak, y_peak


# ============================================================================
# FILE MATCHING
# ============================================================================

def extract_event_key(filename: str) -> Optional[Tuple[str, str]]:
    """
    Extract event key for matching moshits and phels files.

    Examples:
        moshits_Q0_atm01_0014_10PeV_15_001_500m_c001 -> (Q0_atm01_0014_10PeV_15_001, c001)
        phels_to_trace_Q0_atm01_0014_10PeV_15_001_c001 -> (Q0_atm01_0014_10PeV_15_001, c001)

    Args:
        filename: Name of the file (without path)

    Returns:
        Tuple (event_id, clone) or None if pattern doesn't match
    """
    # Extract clone number (c001, c002, ...)
    clone_match = re.search(r'_(c\d+)$', filename)
    clone = clone_match.group(1) if clone_match else None

    # Extract event identifier (Q0_atm01_0014_10PeV_15_001)
    event_match = re.search(r'(Q\d+_atm\d+_\d+_\d+PeV_\d+_\d+)', filename)
    event_id = event_match.group(1) if event_match else None

    if event_id and clone:
        return (event_id, clone)
    return None


def match_event_files(moshits_dir: Path, phels_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Find matching moshits and phels files by event key.

    Args:
        moshits_dir: Directory with moshits files
        phels_dir: Directory with phels files

    Returns:
        List of tuples (moshits_file, phels_file, event_key)
    """
    if not moshits_dir.exists():
        logger.error(f"Moshits directory not found: {moshits_dir}")
        return []

    if not phels_dir.exists():
        logger.error(f"Phels directory not found: {phels_dir}")
        return []

    # Build dictionaries: key -> file path
    moshits_files = {}
    for f in moshits_dir.iterdir():
        if f.is_file() and not f.name.startswith('.'):
            key = extract_event_key(f.name)
            if key:
                moshits_files[key] = f

    phels_files = {}
    for f in phels_dir.iterdir():
        if f.is_file() and not f.name.startswith('.'):
            key = extract_event_key(f.name)
            if key:
                phels_files[key] = f

    # Find matching pairs
    pairs = []
    for key in moshits_files:
        if key in phels_files:
            event_key = f"{key[0]}_{key[1]}"
            pairs.append((moshits_files[key], phels_files[key], event_key))

    logger.info(f"Found {len(pairs)} matching event pairs "
                f"(moshits: {len(moshits_files)}, phels: {len(phels_files)})")

    return pairs


# ============================================================================
# WORKER STATE FOR PARALLEL PROCESSING
# ============================================================================

_worker_geometry: Optional[DetectorGeometry] = None


def _init_comparison_worker(pixel_data_path: Path):
    """Initialize worker process with detector coordinates."""
    global _worker_geometry

    logger.info(f"Initializing comparison worker {mp.current_process().pid}")
    _worker_geometry = load_detector_coordinates(pixel_data_path)
    logger.info(f"Worker {mp.current_process().pid} initialized with {len(_worker_geometry.pix_x)} pixels")


def _compute_ldf_moshits(event_data, which: str = 'pix') -> Tuple[LDF, float, float, float, float]:
    """
    Compute LDF for moshits data using worker's preloaded coordinates.

    Same logic as worker.py process_file but without background subtraction.
    """
    global _worker_geometry

    if event_data is None:
        empty = LDF(I=np.array([]), x=np.array([]), y=np.array([]))
        return empty, 0.0, 0.0, 0.0, 0.0

    ldf = build_ldf(event_data, _worker_geometry, which=which, bg_level=0.0)

    total_I = float(ldf.I.sum())
    if total_I == 0:
        return ldf, 0.0, 0.0, 0.0, 0.0

    x_center = float(np.sqrt((ldf.x**2 * ldf.I).sum() / total_I))
    y_center = float(np.sqrt((ldf.y**2 * ldf.I).sum() / total_I))

    max_idx = int(ldf.I.argmax())
    if ldf.x[max_idx] < 0:
        x_center = -x_center
    if ldf.y[max_idx] < 0:
        y_center = -y_center

    x_peak, y_peak = compute_peak_center(ldf, n_top=N_TOP_PEAKS)

    return ldf, x_center, y_center, x_peak, y_peak


# ============================================================================
# EVENT PAIR PROCESSING
# ============================================================================

def process_event_pair(
    moshits_file: Path,
    phels_file: Path,
    event_key: str,
    min_intensity: float = MIN_TOTAL_INTENSITY
) -> Optional[Dict[str, Any]]:
    """
    Process a single pair of moshits/phels events.

    Args:
        moshits_file: Path to moshits file
        phels_file: Path to phels file
        event_key: Event identifier string
        min_intensity: Minimum total intensity threshold

    Returns:
        Dictionary with comparison results or None if processing fails
    """
    try:
        # Load moshits
        moshits_header = get_first_row(moshits_file)
        moshits_event = load_data(moshits_file)

        # Load phels
        phels_header, phels_data = load_phels_data(phels_file)

        # Check center distance filter
        center_distance = np.sqrt(
            moshits_header.x_center**2 + moshits_header.y_center**2
        )
        if center_distance > MAX_CENTER_DISTANCE:
            logger.debug(f"Event {event_key} filtered: center distance {center_distance:.1f}")
            return None

        # Compute LDF for moshits
        moshits_ldf, m_xc, m_yc, m_xp, m_yp = _compute_ldf_moshits(moshits_event, which=GEOMETRY_TYPE)

        # Compute LDF for phels
        phels_ldf, p_xc, p_yc, p_xp, p_yp = compute_ldf_phels(phels_data)

        # Check intensity threshold
        m_total_I = float(moshits_ldf.I.sum())
        p_total_I = phels_ldf['I'].sum()

        if m_total_I < min_intensity or p_total_I < min_intensity:
            logger.debug(f"Event {event_key} filtered: intensity moshits={m_total_I:.1f}, phels={p_total_I:.1f}")
            return None

        # Check peak distance
        m_peak_dist = np.sqrt(m_xp**2 + m_yp**2)
        p_peak_dist = np.sqrt(p_xp**2 + p_yp**2)

        if m_peak_dist > MAX_PEAK_DISTANCE or p_peak_dist > MAX_PEAK_DISTANCE:
            logger.debug(f"Event {event_key} filtered: peak distance")
            return None

        # Fit LDF for moshits
        moshits_fit = minimize_with_restarts(
            moshits_ldf.I, moshits_ldf.x, moshits_ldf.y,
            m_xp, m_yp,
            backend='scipy', n_restarts=3,
        )

        # Fit LDF for phels
        phels_fit = minimize_with_restarts(
            phels_ldf['I'].to_numpy(), phels_ldf['x'].to_numpy(), phels_ldf['y'].to_numpy(),
            p_xp, p_yp,
            backend='scipy', n_restarts=3,
        )

        if moshits_fit is None or phels_fit is None:
            logger.warning(f"Event {event_key}: optimization failed")
            return None

        # Compute quality metrics
        m_obs = moshits_ldf.I
        m_pred = func(
            moshits_fit.p0, moshits_fit.p1, moshits_fit.p4,
            moshits_fit.s, moshits_fit.x0, moshits_fit.y0,
            moshits_ldf.x, moshits_ldf.y
        )

        p_obs = phels_ldf['I'].to_numpy()
        p_pred = func(
            phels_fit.p0, phels_fit.p1, phels_fit.p4,
            phels_fit.s, phels_fit.x0, phels_fit.y0,
            phels_ldf['x'].values, phels_ldf['y'].values
        )

        # RMSE and R2 for moshits
        m_nonzero = m_obs > 0
        if m_nonzero.sum() > 0:
            m_rmse = np.sqrt(np.mean((m_pred[m_nonzero] - m_obs[m_nonzero])**2))
            m_ss_tot = np.sum((m_obs[m_nonzero] - m_obs[m_nonzero].mean())**2)
            m_ss_res = np.sum((m_pred[m_nonzero] - m_obs[m_nonzero])**2)
            m_r2 = 1 - m_ss_res / m_ss_tot if m_ss_tot > 0 else float('nan')
        else:
            m_rmse, m_r2 = float('nan'), float('nan')

        # RMSE and R2 for phels
        p_nonzero = p_obs > 0
        if p_nonzero.sum() > 0:
            p_rmse = np.sqrt(np.mean((p_pred[p_nonzero] - p_obs[p_nonzero])**2))
            p_ss_tot = np.sum((p_obs[p_nonzero] - p_obs[p_nonzero].mean())**2)
            p_ss_res = np.sum((p_pred[p_nonzero] - p_obs[p_nonzero])**2)
            p_r2 = 1 - p_ss_res / p_ss_tot if p_ss_tot > 0 else float('nan')
        else:
            p_rmse, p_r2 = float('nan'), float('nan')

        # Compute differences
        result = {
            'event_key': event_key,
            # Moshits parameters
            'p0_m': moshits_fit.p0,
            'p1_m': moshits_fit.p1,
            'p4_m': moshits_fit.p4,
            's_m': moshits_fit.s,
            'x0_m': moshits_fit.x0,
            'y0_m': moshits_fit.y0,
            'rmse_m': m_rmse,
            'r2_m': m_r2,
            'I_total_m': m_total_I,
            'I_max_m': m_obs.max(),
            # Phels parameters
            'p0_p': phels_fit.p0,
            'p1_p': phels_fit.p1,
            'p4_p': phels_fit.p4,
            's_p': phels_fit.s,
            'x0_p': phels_fit.x0,
            'y0_p': phels_fit.y0,
            'rmse_p': p_rmse,
            'r2_p': p_r2,
            'I_total_p': p_total_I,
            'I_max_p': p_obs.max(),
            # Differences
            'delta_p0': phels_fit.p0 - moshits_fit.p0,
            'delta_p1': phels_fit.p1 - moshits_fit.p1,
            'delta_p4': phels_fit.p4 - moshits_fit.p4,
            'delta_s': phels_fit.s - moshits_fit.s,
            'delta_x0': phels_fit.x0 - moshits_fit.x0,
            'delta_y0': phels_fit.y0 - moshits_fit.y0,
            'delta_rmse': p_rmse - m_rmse,
            'delta_r2': p_r2 - m_r2,
            # Center distance
            'center_dist': np.sqrt(
                (phels_fit.x0 - moshits_fit.x0)**2 +
                (phels_fit.y0 - moshits_fit.y0)**2
            ),
            # Input parameters
            'Rc_snow': center_distance,
            'h': moshits_header.h,
        }

        logger.debug(
            f"Event {event_key}: p0_m={moshits_fit.p0:.3f}, p0_p={phels_fit.p0:.3f}, "
            f"delta={result['delta_p0']:.3f}"
        )

        return result

    except Exception as e:
        logger.error(f"Error processing event pair {event_key}: {e}")
        return None


def _process_pair_wrapper(args):
    """Wrapper for parallel processing."""
    moshits_file, phels_file, event_key = args
    return process_event_pair(moshits_file, phels_file, event_key)


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comparison_plots(results_df: pd.DataFrame, output_dir: Path, particle_type: str):
    """
    Create comparison visualization plots.

    Args:
        results_df: DataFrame with comparison results
        output_dir: Directory to save plots
        particle_type: Particle type for plot titles (p, N, Fe)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    params = ['p0', 'p1', 'p4', 's', 'x0', 'y0']

    # 1. Scatter plots: moshits vs phels for each parameter
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        m_col = f'{param}_m'
        p_col = f'{param}_p'

        if m_col in results_df.columns and p_col in results_df.columns:
            x = results_df[m_col].values
            y = results_df[p_col].values

            # Filter NaN
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]

            if len(x) > 0:
                ax.scatter(x, y, alpha=0.5, s=20)

                # Add y=x line
                lims = [min(x.min(), y.min()), max(x.max(), y.max())]
                ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')

                # Correlation coefficient
                if len(x) > 1:
                    corr = np.corrcoef(x, y)[0, 1]
                    ax.set_title(f'{param} (r={corr:.3f})')
                else:
                    ax.set_title(param)

                ax.set_xlabel(f'{param} (moshits)')
                ax.set_ylabel(f'{param} (phels)')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

    plt.suptitle(f'{particle_type} - Parameter Comparison: moshits vs phels', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / f'param_scatter_{particle_type}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

    # 2. Histograms of differences
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        delta_col = f'delta_{param}'

        if delta_col in results_df.columns:
            values = results_df[delta_col].dropna().values

            if len(values) > 0:
                ax.hist(values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                ax.axvline(0, color='red', linewidth=2, linestyle='--')
                ax.axvline(values.mean(), color='green', linewidth=2, linestyle='-',
                          label=f'Mean: {values.mean():.4f}')
                ax.set_xlabel(f'Delta {param} (phels - moshits)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Delta {param}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

    plt.suptitle(f'{particle_type} - Parameter Differences Distribution', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / f'delta_histograms_{particle_type}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

    # 3. RMSE and R2 comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # RMSE scatter
    ax = axes[0]
    if 'rmse_m' in results_df.columns and 'rmse_p' in results_df.columns:
        x = results_df['rmse_m'].values
        y = results_df['rmse_p'].values
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        if len(x) > 0:
            ax.scatter(x, y, alpha=0.5, s=20)
            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')
            if len(x) > 1:
                corr = np.corrcoef(x, y)[0, 1]
                ax.set_title(f'RMSE (r={corr:.3f})')
            ax.set_xlabel('RMSE (moshits)')
            ax.set_ylabel('RMSE (phels)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # R2 scatter
    ax = axes[1]
    if 'r2_m' in results_df.columns and 'r2_p' in results_df.columns:
        x = results_df['r2_m'].values
        y = results_df['r2_p'].values
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        if len(x) > 0:
            ax.scatter(x, y, alpha=0.5, s=20)
            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')
            if len(x) > 1:
                corr = np.corrcoef(x, y)[0, 1]
                ax.set_title(f'R2 (r={corr:.3f})')
            ax.set_xlabel('R2 (moshits)')
            ax.set_ylabel('R2 (phels)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle(f'{particle_type} - Fit Quality Comparison', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / f'quality_comparison_{particle_type}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

    # 4. Summary statistics
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    summary_lines = [
        f"COMPARISON SUMMARY - {particle_type}",
        "=" * 50,
        f"Total events: {len(results_df)}",
        "",
        "Parameter Differences (phels - moshits):",
        "-" * 40,
    ]

    for param in params:
        delta_col = f'delta_{param}'
        if delta_col in results_df.columns:
            vals = results_df[delta_col].dropna()
            if len(vals) > 0:
                summary_lines.append(
                    f"  {param:5s}: mean={vals.mean():+.4f}, std={vals.std():.4f}, "
                    f"median={vals.median():+.4f}"
                )

    summary_lines.extend([
        "",
        "Quality Metrics:",
        "-" * 40,
    ])

    for metric in ['rmse', 'r2']:
        m_col = f'{metric}_m'
        p_col = f'{metric}_p'
        if m_col in results_df.columns and p_col in results_df.columns:
            m_vals = results_df[m_col].dropna()
            p_vals = results_df[p_col].dropna()
            if len(m_vals) > 0 and len(p_vals) > 0:
                summary_lines.append(
                    f"  {metric.upper():5s} moshits: mean={m_vals.mean():.4f}, std={m_vals.std():.4f}"
                )
                summary_lines.append(
                    f"  {metric.upper():5s} phels:   mean={p_vals.mean():.4f}, std={p_vals.std():.4f}"
                )

    if 'center_dist' in results_df.columns:
        cd = results_df['center_dist'].dropna()
        if len(cd) > 0:
            summary_lines.extend([
                "",
                f"Center distance: mean={cd.mean():.2f} mm, max={cd.max():.2f} mm"
            ])

    ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    fig.savefig(output_dir / f'summary_{particle_type}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Created comparison plots in {output_dir}")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_particle_type(
    particle_type: str,
    moshits_root: Path,
    phels_root: Path,
    pixel_data_path: Path,
    output_dir: Path,
    workers: int,
    files_limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Process all events for a single particle type.

    Args:
        particle_type: 'p', 'N', or 'Fe'
        moshits_root: Root directory for moshits files
        phels_root: Root directory for phels files
        pixel_data_path: Path to detector geometry file
        output_dir: Output directory for results
        workers: Number of worker processes
        files_limit: Optional limit on number of files

    Returns:
        DataFrame with comparison results
    """
    moshits_dir = moshits_root / f'moshits_{particle_type}'
    phels_dir = phels_root / f'phels_{particle_type}'

    logger.info(f"Processing {particle_type}: moshits={moshits_dir}, phels={phels_dir}")

    # Find matching files
    pairs = match_event_files(moshits_dir, phels_dir)

    if not pairs:
        logger.warning(f"No matching files found for {particle_type}")
        return pd.DataFrame()

    if files_limit and files_limit > 0:
        pairs = pairs[:files_limit]
        logger.info(f"Limited to {len(pairs)} pairs")

    # Process pairs in parallel
    results = []

    with mp.Pool(
        processes=workers,
        initializer=_init_comparison_worker,
        initargs=(pixel_data_path,)
    ) as pool:
        for idx, result in enumerate(pool.imap_unordered(_process_pair_wrapper, pairs), 1):
            if result is not None:
                results.append(result)

            if idx % 50 == 0 or idx == len(pairs):
                logger.info(f"Processed {idx}/{len(pairs)} pairs ({len(results)} successful)")

    if not results:
        logger.warning(f"No successful results for {particle_type}")
        return pd.DataFrame()

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save CSV
    csv_path = output_dir / f'comparison_{particle_type}_results.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(results_df)} results to {csv_path}")

    # Create plots
    create_comparison_plots(results_df, output_dir, particle_type)

    return results_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare moshits and phels LDF approximations'
    )
    parser.add_argument(
        '--moshits-root', type=Path, default=DEFAULT_MOSHITS_ROOT,
        help='Root directory with moshits_p, moshits_N, moshits_Fe folders'
    )
    parser.add_argument(
        '--phels-root', type=Path, default=DEFAULT_PHELS_ROOT,
        help='Root directory with phels_p, phels_N, phels_Fe folders'
    )
    parser.add_argument(
        '--pixel-path', type=Path, default=DEFAULT_PIXEL_DATA_PATH,
        help='Path to detector geometry file'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
        help='Output directory for results and plots'
    )
    parser.add_argument(
        '--workers', type=int, default=max(1, (os.cpu_count() or 4) - 1),
        help='Number of worker processes'
    )
    parser.add_argument(
        '--files-limit', type=int, default=None,
        help='Limit number of files per particle type'
    )
    parser.add_argument(
        '--particles', type=str, nargs='+', default=['p'],
        help='Particle types to process (p, N, Fe)'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Starting moshits/phels comparison")
    logger.info("=" * 60)
    logger.info(f"Moshits root: {args.moshits_root}")
    logger.info(f"Phels root: {args.phels_root}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Particles: {args.particles}")

    # Validate inputs
    if not args.pixel_path.exists():
        logger.error(f"Pixel data file not found: {args.pixel_path}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each particle type
    all_results = {}
    for particle in args.particles:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing particle type: {particle}")
        logger.info(f"{'='*60}")

        results_df = process_particle_type(
            particle,
            args.moshits_root,
            args.phels_root,
            args.pixel_path,
            args.output_dir,
            args.workers,
            args.files_limit
        )

        all_results[particle] = results_df

    logger.info("\n" + "=" * 60)
    logger.info("Comparison completed successfully")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    try:
        mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass

    exit(main())
