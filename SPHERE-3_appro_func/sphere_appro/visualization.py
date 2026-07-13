# sphere_appro/visualization.py
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

PARAM_NAMES = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'R_ch', 'sw', 'x0', 'y0']
PARAM_INDICES = list(range(11))


def plot_parameter_distributions(results: np.ndarray, particle_type: str,
                                  output_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_params = len(PARAM_NAMES)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, (name, col_idx) in enumerate(zip(PARAM_NAMES, PARAM_INDICES)):
        ax = axes[i]
        data = results[:, col_idx]
        valid = np.isfinite(data)
        if valid.sum() > 0:
            ax.hist(data[valid], bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(f'{name} ({particle_type})')
        ax.set_xlabel(name)
        ax.set_ylabel('Count')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / f'params_{particle_type}.png', dpi=150)
    plt.close()


def plot_fit_quality(results: np.ndarray, particle_type: str,
                     output_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chi2_ndf = results[:, 12]
    mean_abs_d = results[:, 14]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # chi2/ndf histogram
    valid_chi2 = np.isfinite(chi2_ndf)
    if valid_chi2.sum() > 0:
        axes[0].hist(chi2_ndf[valid_chi2], bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_title(f'chi2/ndf ({particle_type})')
    axes[0].set_xlabel('chi2/ndf')

    # mean_abs_d histogram
    valid_mad = np.isfinite(mean_abs_d)
    if valid_mad.sum() > 0:
        axes[1].hist(mean_abs_d[valid_mad], bins=30, alpha=0.7, edgecolor='black',
                     color='green')
    axes[1].set_title(f'mean |d| ({particle_type})')
    axes[1].set_xlabel('mean |d|')

    # chi2/ndf vs mean_abs_d scatter
    valid = valid_chi2 & valid_mad
    if valid.sum() > 0:
        axes[2].scatter(chi2_ndf[valid], mean_abs_d[valid], alpha=0.3, s=10)
    axes[2].set_title(f'chi2/ndf vs mean |d| ({particle_type})')
    axes[2].set_xlabel('chi2/ndf')
    axes[2].set_ylabel('mean |d|')

    plt.tight_layout()
    plt.savefig(output_dir / f'quality_{particle_type}.png', dpi=150)
    plt.close()


def plot_criterion_histograms(criteria: dict, borders: dict,
                               output_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = len(criteria)
    fig, axes = plt.subplots(1, max(n_pairs, 1), figsize=(7 * max(n_pairs, 1), 5))
    if n_pairs == 1:
        axes = [axes]

    for ax, (pair_name, (data_a, data_b)) in zip(axes, criteria.items()):
        valid_a = np.isfinite(data_a)
        valid_b = np.isfinite(data_b)
        if valid_a.sum() > 0:
            ax.hist(data_a[valid_a], bins=50, alpha=0.5, label=pair_name.split('-')[0])
        if valid_b.sum() > 0:
            ax.hist(data_b[valid_b], bins=50, alpha=0.5, label=pair_name.split('-')[1])
        if pair_name in borders:
            ax.axvline(borders[pair_name], color='red', linestyle='--',
                      label=f'border={borders[pair_name]:.3f}')
        ax.set_title(f'Criterion: {pair_name}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_visualizations(results: np.ndarray, folder: str,
                           particle_type: str, output_dir: Path = None):
    if output_dir is None:
        output_dir = Path('debug_plots')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_parameter_distributions(results, particle_type, output_dir)
        plot_fit_quality(results, particle_type, output_dir)
        logger.info('Saved visualizations for %s to %s', particle_type, output_dir)
    except Exception as e:
        logger.warning('Visualization failed for %s: %s', particle_type, e)
