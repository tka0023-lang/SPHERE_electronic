import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Import visualization module
try:
    from minuit_visualization import (
        visualize_minuit_optimization,
        plot_parameter_distributions,
        plot_parameter_correlations,
        plot_fit_residuals,
        plot_minuit_summary
    )
except ImportError:
    print("Error: Could not import minuit_visualization module")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
DEFAULT_OUTPUT_DIR = 'debug_plots'
DEFAULT_CSV_FILE = 'moshits_p_params.csv'

# Number of samples to analyze
N_SAMPLES = 50


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_sample_results(csv_file: str, n_samples: int = None) -> pd.DataFrame:
    """
    Load sample minuit optimization results from CSV file.

    Args:
        csv_file: Path to CSV file with results
        n_samples: Optional limit on number of samples

    Returns:
        DataFrame with results
    """
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return None

    try:
        df = pd.read_csv(csv_file)
        if n_samples is not None:
            df = df.head(n_samples)
        logger.info(f"Loaded {len(df)} samples from {csv_file}")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return None


def visualize_parameter_statistics(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR):
    """
    Visualize parameter statistics across multiple events.

    Args:
        df: DataFrame with optimization results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract parameter columns
    param_cols = ['p0', 'p1', 'p2', 'p3', 'p4', 's', 'x0', 'y0']
    available_cols = [col for col in param_cols if col in df.columns]

    logger.info(f"Found {len(available_cols)} parameter columns: {available_cols}")

    # Create figure with parameter statistics
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=100)
    axes = axes.flatten()

    for idx, param in enumerate(available_cols):
        ax = axes[idx]
        values = df[param].values

        ax.hist(values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(values.mean(), color='red', linewidth=2, linestyle='--', label=f'Mean: {values.mean():.4f}')
        ax.axvline(np.median(values), color='green', linewidth=2, linestyle='--', label=f'Median: {np.median(values):.4f}')

        ax.set_xlabel(param)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Parameter {param} Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Print statistics
        logger.info(f"{param}: mean={values.mean():.4f}, std={values.std():.4f}, "
                   f"min={values.min():.4f}, max={values.max():.4f}")

    # Hide unused subplots
    for idx in range(len(available_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Parameter Statistics Across All Events', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'parameter_statistics.png')
    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
    logger.info(f"Saved parameter statistics plot to {plot_path}")
    plt.close(fig)


def visualize_fit_quality(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR):
    """
    Visualize fit quality metrics (RMSE and R²).

    Args:
        df: DataFrame with optimization results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    axes = axes.flatten()

    # RMSE distribution
    if 'rmse' in df.columns:
        ax = axes[0]
        rmse_values = df['rmse'].values
        ax.hist(rmse_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(rmse_values.mean(), color='red', linewidth=2, linestyle='--',
                  label=f'Mean: {rmse_values.mean():.4f}')
        ax.set_xlabel('RMSE')
        ax.set_ylabel('Frequency')
        ax.set_title('RMSE Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        logger.info(f"RMSE: mean={rmse_values.mean():.4f}, std={rmse_values.std():.4f}")

    # R² distribution
    if 'r2' in df.columns:
        ax = axes[1]
        r2_values = df['r2'].values
        ax.hist(r2_values, bins=30, edgecolor='black', alpha=0.7, color='seagreen')
        ax.axvline(r2_values.mean(), color='red', linewidth=2, linestyle='--',
                  label=f'Mean: {r2_values.mean():.4f}')
        ax.set_xlabel('R²')
        ax.set_ylabel('Frequency')
        ax.set_title('R² Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        logger.info(f"R²: mean={r2_values.mean():.4f}, std={r2_values.std():.4f}")

    # RMSE vs R²
    if 'rmse' in df.columns and 'r2' in df.columns:
        ax = axes[2]
        ax.scatter(df['rmse'].values, df['r2'].values, alpha=0.5, s=20)
        ax.set_xlabel('RMSE')
        ax.set_ylabel('R²')
        ax.set_title('RMSE vs R²')
        ax.grid(True, alpha=0.3)

    # Summary statistics table
    ax = axes[3]
    ax.axis('off')

    summary_text = "FIT QUALITY SUMMARY\n" + "="*40 + "\n\n"
    if 'rmse' in df.columns:
        rmse_values = df['rmse'].values
        summary_text += f"RMSE\n  Mean: {rmse_values.mean():.6f}\n  Std:  {rmse_values.std():.6f}\n  Min:  {rmse_values.min():.6f}\n  Max:  {rmse_values.max():.6f}\n\n"

    if 'r2' in df.columns:
        r2_values = df['r2'].values
        summary_text += f"R²\n  Mean: {r2_values.mean():.6f}\n  Std:  {r2_values.std():.6f}\n  Min:  {r2_values.min():.6f}\n  Max:  {r2_values.max():.6f}\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           verticalalignment='top', fontfamily='monospace', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.suptitle('Fit Quality Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'fit_quality.png')
    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
    logger.info(f"Saved fit quality plot to {plot_path}")
    plt.close(fig)


def visualize_single_event(df: pd.DataFrame, event_idx: int = 0, output_dir: str = DEFAULT_OUTPUT_DIR):
    """
    Visualize optimization result for a single event.

    Args:
        df: DataFrame with optimization results
        event_idx: Index of event to visualize
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    if event_idx >= len(df):
        logger.error(f"Event index {event_idx} out of range (max: {len(df)-1})")
        return

    event_row = df.iloc[event_idx]
    logger.info(f"Visualizing event {event_idx}")

    # Extract minuit result dictionary
    minuit_result = {
        'p0': float(event_row['p0']),
        'p1': float(event_row['p1']),
        'p2': float(event_row['p2']),
        'p3': float(event_row['p3']),
        'p4': float(event_row['p4']),
        's': float(event_row['s']),
        'x0': float(event_row['x0']),
        'y0': float(event_row['y0']),
        'fcn': float(event_row.get('fcn', np.nan)) if 'fcn' in event_row else np.nan
    }

    # Create parameter distribution plot
    try:
        fig = plot_parameter_distributions(
            {k: v for k, v in minuit_result.items() if k != 'fcn'},
            title=f"Event {event_idx} - Parameter Values",
            save_path=os.path.join(output_dir, f'event_{event_idx}_parameters.png')
        )
        logger.info(f"Saved event {event_idx} parameter plot")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating event parameter plot: {e}")


def create_event_comparison(df: pd.DataFrame, n_events: int = 4, output_dir: str = DEFAULT_OUTPUT_DIR):
    """
    Create comparison plots for multiple events.

    Args:
        df: DataFrame with optimization results
        n_events: Number of events to compare
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    import matplotlib.pyplot as plt

    n_events = min(n_events, len(df))
    n_params = 8

    fig, axes = plt.subplots(n_events, n_params, figsize=(16, 3*n_events), dpi=100)
    if n_events == 1:
        axes = axes.reshape(1, -1)

    param_cols = ['p0', 'p1', 'p2', 'p3', 'p4', 's', 'x0', 'y0']
    available_cols = [col for col in param_cols if col in df.columns]

    for event_i in range(n_events):
        for param_j, param in enumerate(available_cols):
            ax = axes[event_i, param_j] if n_events > 1 else axes[param_j]
            value = df.iloc[event_i][param]

            ax.bar([0], [value], color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_ylabel('Value')

            if event_i == 0:
                ax.set_title(param)

            ax.text(0, value, f'{value:.4f}', ha='center', va='bottom', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Event Comparison - Parameter Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'event_comparison.png')
    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
    logger.info(f"Saved event comparison plot to {plot_path}")
    plt.close(fig)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main debug script execution."""
    logger.info("="*70)
    logger.info("MINUIT VISUALIZATION DEBUG SCRIPT")
    logger.info("="*70)

    # Create output directory
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    # Load sample data
    df = load_sample_results(DEFAULT_CSV_FILE, n_samples=N_SAMPLES)
    if df is None:
        logger.error("Failed to load sample data")
        return

    logger.info(f"Loaded {len(df)} events from {DEFAULT_CSV_FILE}")

    # Create visualizations
    try:
        logger.info("Creating parameter statistics plots...")
        visualize_parameter_statistics(df, DEFAULT_OUTPUT_DIR)

        logger.info("Creating fit quality plots...")
        visualize_fit_quality(df, DEFAULT_OUTPUT_DIR)

        logger.info("Creating single event visualization...")
        visualize_single_event(df, event_idx=0, output_dir=DEFAULT_OUTPUT_DIR)
        visualize_single_event(df, event_idx=1, output_dir=DEFAULT_OUTPUT_DIR)

        logger.info("Creating event comparison plots...")
        create_event_comparison(df, n_events=4, output_dir=DEFAULT_OUTPUT_DIR)

    except Exception as e:
        logger.error(f"Error during visualization: {e}", exc_info=True)
        return

    logger.info("="*70)
    logger.info(f"All visualization plots saved to '{DEFAULT_OUTPUT_DIR}/'")
    logger.info("="*70)


if __name__ == '__main__':
    main()