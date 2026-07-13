"""Separability map: combine criterion and ML results into unified analysis."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_separability_table(criterion_path, ml_metrics_path):
    """Build unified separability table from criterion and ML results.

    Either path can be None — the corresponding columns will be NaN.
    """
    crit_df = None
    ml_slices = None

    if criterion_path is not None:
        crit_df = pd.read_parquet(criterion_path)
        crit_df = crit_df[crit_df['energy'] != 'all'].copy()

    if ml_metrics_path is not None:
        with open(ml_metrics_path) as f:
            ml_metrics = json.load(f)
        ml_slices = pd.DataFrame(ml_metrics.get('per_slice', []))
        if not ml_slices.empty:
            model_abbrev = {'random_forest': 'rf', 'xgboost': 'xgb', 'lightgbm': 'lgbm'}
            for model, abbr in model_abbrev.items():
                if model in ml_slices.columns:
                    ml_slices[f'ml_accuracy_{abbr}'] = (
                        ml_slices[model].apply(lambda x: x['accuracy'] if isinstance(x, dict) else np.nan)
                    )
                    ml_slices.drop(columns=[model], inplace=True)

    if crit_df is not None and ml_slices is not None and not ml_slices.empty:
        crit_renamed = crit_df.rename(columns={
            'error_pN': 'criterion_error_pN',
            'error_NFe': 'criterion_error_NFe',
        })[['energy', 'angle', 'height', 'criterion_error_pN', 'criterion_error_NFe']]
        table = pd.merge(crit_renamed, ml_slices, on=['energy', 'angle', 'height'], how='outer')
    elif crit_df is not None:
        table = crit_df.rename(columns={
            'error_pN': 'criterion_error_pN',
            'error_NFe': 'criterion_error_NFe',
        })[['energy', 'angle', 'height', 'criterion_error_pN', 'criterion_error_NFe']]
        table['ml_accuracy_rf'] = np.nan
        table['ml_accuracy_xgb'] = np.nan
        table['ml_accuracy_lgbm'] = np.nan
    elif ml_slices is not None and not ml_slices.empty:
        table = ml_slices.copy()
        table['criterion_error_pN'] = np.nan
        table['criterion_error_NFe'] = np.nan
    else:
        table = pd.DataFrame(columns=[
            'energy', 'angle', 'height',
            'criterion_error_pN', 'criterion_error_NFe',
            'ml_accuracy_rf', 'ml_accuracy_xgb', 'ml_accuracy_lgbm',
        ])

    return table


def plot_separability(table, output_dir):
    """Generate heatmaps and trend plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if table.empty:
        logger.warning('Empty separability table, skipping plots')
        return

    ml_cols = [c for c in table.columns if c.startswith('ml_accuracy_')]
    if ml_cols:
        table['ml_accuracy_best'] = table[ml_cols].max(axis=1)

    for height in table['height'].unique():
        subset = table[table['height'] == height]
        if subset.empty:
            continue

        if 'ml_accuracy_best' in subset.columns and subset['ml_accuracy_best'].notna().any():
            pivot = subset.pivot_table(index='angle', columns='energy', values='ml_accuracy_best')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, vmin=0.33, vmax=1.0)
            ax.set_title(f'ML Accuracy (best model), height={height}m')
            fig.tight_layout()
            fig.savefig(output_dir / f'heatmap_ml_accuracy_h{height}.png', dpi=150)
            plt.close(fig)

        if subset['criterion_error_pN'].notna().any():
            for err_col, pair_label in [('criterion_error_pN', 'p-N'), ('criterion_error_NFe', 'N-Fe')]:
                pivot = subset.pivot_table(index='angle', columns='energy', values=err_col)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax, vmin=0.0, vmax=0.5)
                ax.set_title(f'Criterion Error ({pair_label}), height={height}m')
                fig.tight_layout()
                fig.savefig(output_dir / f'heatmap_criterion_{pair_label.replace("-","_")}_h{height}.png', dpi=150)
                plt.close(fig)

    if 'ml_accuracy_best' in table.columns:
        trend_energy = table.groupby('energy')['ml_accuracy_best'].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        trend_energy.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_ylabel('Mean ML Accuracy')
        ax.set_title('ML Accuracy vs Energy')
        ax.set_ylim(0.33, 1.0)
        fig.tight_layout()
        fig.savefig(output_dir / 'trend_accuracy_vs_energy.png', dpi=150)
        plt.close(fig)

        trend_angle = table.groupby('angle')['ml_accuracy_best'].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        trend_angle.plot(kind='bar', ax=ax, color='darkorange')
        ax.set_ylabel('Mean ML Accuracy')
        ax.set_title('ML Accuracy vs Angle')
        ax.set_ylim(0.33, 1.0)
        fig.tight_layout()
        fig.savefig(output_dir / 'trend_accuracy_vs_angle.png', dpi=150)
        plt.close(fig)

    logger.info('Plots saved to %s', output_dir)


def run_separability_analysis(input_dir, output_dir=None):
    """Full separability analysis: load results, build table, plot."""
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    output_dir = Path(output_dir)

    crit_path = input_dir / 'criterion_results.parquet'
    ml_path = input_dir / 'ml_metrics.json'

    table = build_separability_table(
        str(crit_path) if crit_path.exists() else None,
        str(ml_path) if ml_path.exists() else None,
    )

    table.to_parquet(output_dir / 'separability_map.parquet', index=False)
    logger.info('Separability table saved (%d rows)', len(table))

    plot_separability(table, output_dir)
    return table


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='Separability map for SPHERE-3')
    parser.add_argument('input_dir', help='Directory with criterion_results.parquet and ml_metrics.json')
    parser.add_argument('-o', '--output-dir', default=None, help='Output directory (default: same as input)')
    args = parser.parse_args()

    run_separability_analysis(args.input_dir, args.output_dir)
