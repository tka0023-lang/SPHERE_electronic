import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib import cm
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Plotting parameters
DEFAULT_FIGSIZE = (14, 10)
DEFAULT_DPI = 100
HISTOGRAM_BINS = 30
SCATTER_ALPHA = 0.5
SCATTER_SIZE = 20

# Color scheme
COLOR_BEST_FIT = 'red'
COLOR_INITIAL = 'blue'
COLOR_CONTOUR = 'gray'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_minuit_info(minuit_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract optimization information from Minuit result dictionary.

    Args:
        minuit_result: Dictionary with Minuit optimization results

    Returns:
        Dictionary with extracted information:
            - parameters: Dict of parameter values
            - errors: Dict of parameter errors (if available)
            - fcn_value: Final function value
            - covariance: Covariance matrix (if available)
    """
    info = {
        'parameters': {},
        'errors': {},
        'fcn_value': minuit_result.get('fcn'),
        'covariance': None
    }

    # Extract parameter names and values
    param_names = [k for k in minuit_result.keys() if k != 'fcn']
    for name in param_names:
        info['parameters'][name] = minuit_result[name]

    return info


def calculate_parameter_ranges(param_values: np.ndarray,
                              error_values: Optional[np.ndarray] = None,
                              nsigma: float = 3.0) -> Tuple[float, float]:
    """
    Calculate reasonable axis ranges for parameter visualization.

    Args:
        param_values: Array of parameter values
        error_values: Optional array of parameter errors
        nsigma: Number of standard deviations for range calculation

    Returns:
        Tuple (min_val, max_val) for axis range
    """
    if error_values is not None and np.any(error_values > 0):
        mean_err = np.nanmean(error_values[error_values > 0])
        min_val = np.nanmin(param_values) - nsigma * mean_err
        max_val = np.nanmax(param_values) + nsigma * mean_err
    else:
        range_val = (np.nanmax(param_values) - np.nanmin(param_values)) * 0.1
        min_val = np.nanmin(param_values) - range_val
        max_val = np.nanmax(param_values) + range_val

    return float(min_val), float(max_val)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_parameter_distributions(param_dict: Dict[str, float],
                                error_dict: Optional[Dict[str, float]] = None,
                                title: str = "Parameter Distribution",
                                figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create histogram plots for parameter values with optional error bars.

    Args:
        param_dict: Dictionary {param_name: value}
        error_dict: Optional dictionary {param_name: error}
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    n_params = len(param_dict)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=DEFAULT_DPI)
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    param_names = list(param_dict.keys())
    for idx, (ax, param_name) in enumerate(zip(axes, param_names)):
        param_value = param_dict[param_name]
        error_value = error_dict.get(param_name) if error_dict else None

        # Create histogram
        ax.axvline(param_value, color=COLOR_BEST_FIT, linewidth=2, label='Optimized value')

        if error_value is not None and error_value > 0:
            ax.axvspan(param_value - error_value, param_value + error_value,
                      alpha=0.3, color=COLOR_BEST_FIT, label=f'±1σ error')
            ax.text(0.95, 0.95, f'Value: {param_value:.4f}\nError: {error_value:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   horizontalalignment='right', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.95, 0.95, f'Value: {param_value:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   horizontalalignment='right', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(param_name)
        ax.set_ylabel('Value')
        ax.set_title(param_name)
        if error_value is not None:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(param_names), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info(f"Saved parameter distribution plot to {save_path}")

    return fig


def plot_parameter_correlations(param_dict: Dict[str, float],
                               error_dict: Optional[Dict[str, float]] = None,
                               title: str = "Parameter Correlations",
                               figsize: Optional[Tuple[int, int]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create 2D correlation plots between parameters.

    Args:
        param_dict: Dictionary {param_name: value}
        error_dict: Optional dictionary {param_name: error}
        title: Plot title
        figsize: Figure size (auto-calculated if None)
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    param_names = list(param_dict.keys())
    n_params = len(param_names)

    if n_params < 2:
        logger.warning("Need at least 2 parameters for correlation plot")
        return None

    if figsize is None:
        figsize = (4 * n_params, 4 * n_params)

    fig, axes = plt.subplots(n_params, n_params, figsize=figsize, dpi=DEFAULT_DPI)

    for i, param_i in enumerate(param_names):
        for j, param_j in enumerate(param_names):
            ax = axes[i, j]

            if i == j:
                # Diagonal: show parameter value and error
                val_i = param_dict[param_i]
                err_i = error_dict.get(param_i) if error_dict else None

                ax.axvline(val_i, color=COLOR_BEST_FIT, linewidth=2)
                if err_i is not None and err_i > 0:
                    ax.axvspan(val_i - err_i, val_i + err_i,
                              alpha=0.3, color=COLOR_BEST_FIT)
                ax.set_ylabel('Frequency')
                ax.text(0.5, 0.5, f'{param_i}\n{val_i:.4f}',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            else:
                # Off-diagonal: show 2D scatter plot
                ax.scatter([param_dict[param_j]], [param_dict[param_i]],
                          s=SCATTER_SIZE*5, color=COLOR_BEST_FIT, zorder=3)

                # Add error ellipse if available
                if error_dict:
                    err_i = error_dict.get(param_i, 0)
                    err_j = error_dict.get(param_j, 0)
                    if err_i > 0 and err_j > 0:
                        ellipse = Ellipse((param_dict[param_j], param_dict[param_i]),
                                        2 * err_j, 2 * err_i,
                                        fill=False, edgecolor=COLOR_BEST_FIT,
                                        linewidth=2, alpha=0.5)
                        ax.add_patch(ellipse)

            ax.set_xlabel(param_j if i == n_params - 1 else '')
            ax.set_ylabel(param_i if j == 0 else '')
            ax.grid(True, alpha=0.3)

            if i > 0:
                ax.set_xticklabels([])
            if j < n_params - 1:
                ax.set_yticklabels([])

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info(f"Saved parameter correlation plot to {save_path}")

    return fig


def plot_fit_residuals(observed: np.ndarray,
                      predicted: np.ndarray,
                      x_coords: Optional[np.ndarray] = None,
                      y_coords: Optional[np.ndarray] = None,
                      title: str = "Fit Residuals Analysis",
                      figsize: Tuple[int, int] = (14, 10),
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive residual analysis plots.

    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        x_coords: Optional x coordinates for spatial residual plot
        y_coords: Optional y coordinates for spatial residual plot
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    residuals = observed - predicted

    # Calculate statistics
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    r2 = 1 - np.sum(residuals**2) / np.sum((observed - np.mean(observed))**2)

    n_plots = 3 if (x_coords is not None and y_coords is not None) else 2
    fig, axes = plt.subplots(2, n_plots if n_plots <= 2 else 2,
                             figsize=figsize, dpi=DEFAULT_DPI)
    axes = axes.flatten()

    # Plot 1: Observed vs Predicted
    ax = axes[0]
    ax.scatter(observed, predicted, alpha=SCATTER_ALPHA, s=SCATTER_SIZE)
    min_val = min(observed.min(), predicted.min())
    max_val = max(observed.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('Observed vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Residuals histogram
    ax = axes[1]
    ax.hist(residuals, bins=HISTOGRAM_BINS, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linewidth=2, linestyle='--')
    ax.set_xlabel('Residual Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution')
    ax.text(0.98, 0.97, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
           transform=ax.transAxes, verticalalignment='top',
           horizontalalignment='right', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3)

    # Plot 3: Residuals vs Index
    ax = axes[2]
    ax.scatter(range(len(residuals)), residuals, alpha=SCATTER_ALPHA, s=SCATTER_SIZE)
    ax.axhline(0, color='r', linewidth=2, linestyle='--')
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Index')
    ax.grid(True, alpha=0.3)

    # Plot 4: Spatial residuals (if coordinates provided)
    if x_coords is not None and y_coords is not None:
        ax = axes[3]
        scatter = ax.scatter(x_coords, y_coords, c=residuals, cmap='RdBu_r',
                           s=SCATTER_SIZE, alpha=SCATTER_ALPHA)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Spatial Residual Distribution')
        plt.colorbar(scatter, ax=ax, label='Residual')
        ax.grid(True, alpha=0.3)
    else:
        axes[3].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info(f"Saved residual analysis plot to {save_path}")

    return fig


def plot_function_landscape(func, param_ranges: Dict[str, Tuple[float, float]],
                           fixed_params: Optional[Dict[str, float]] = None,
                           optimal_params: Optional[Dict[str, float]] = None,
                           title: str = "Function Landscape",
                           figsize: Tuple[int, int] = (12, 10),
                           save_path: Optional[str] = None,
                           n_points: int = 50) -> plt.Figure:
    """
    Visualize 2D function landscape for two-parameter scan.

    Args:
        func: Function to evaluate func(param_dict) -> float
        param_ranges: Dictionary {param_name: (min, max)} for 2 parameters
        fixed_params: Dictionary of other fixed parameters
        optimal_params: Dictionary with optimal parameter values
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        n_points: Number of points per dimension

    Returns:
        Matplotlib Figure object
    """
    if len(param_ranges) != 2:
        logger.error("Function landscape requires exactly 2 parameters")
        return None

    param_names = list(param_ranges.keys())
    param1_name, param2_name = param_names[0], param_names[1]
    (p1_min, p1_max), (p2_min, p2_max) = param_ranges[param1_name], param_ranges[param2_name]

    # Create grid
    p1_values = np.linspace(p1_min, p1_max, n_points)
    p2_values = np.linspace(p2_min, p2_max, n_points)
    p1_grid, p2_grid = np.meshgrid(p1_values, p2_values)

    # Evaluate function on grid
    z_grid = np.zeros_like(p1_grid)
    for i in range(n_points):
        for j in range(n_points):
            params = fixed_params.copy() if fixed_params else {}
            params[param1_name] = p1_grid[i, j]
            params[param2_name] = p2_grid[i, j]
            try:
                z_grid[i, j] = func(params)
            except:
                z_grid[i, j] = np.nan

    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    # Create contour plot
    levels = np.linspace(np.nanmin(z_grid), np.nanmin(z_grid) + 0.1 * np.nanmax(z_grid), 20)
    contour = ax.contourf(p1_grid, p2_grid, z_grid, levels=levels, cmap='viridis')
    ax.contour(p1_grid, p2_grid, z_grid, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    # Mark optimal point
    if optimal_params:
        opt_p1 = optimal_params.get(param1_name)
        opt_p2 = optimal_params.get(param2_name)
        if opt_p1 is not None and opt_p2 is not None:
            ax.plot(opt_p1, opt_p2, 'r*', markersize=15, label='Optimum', markeredgecolor='white', markeredgewidth=2)

    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_title(title)
    plt.colorbar(contour, ax=ax, label='Function Value')
    if optimal_params:
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info(f"Saved function landscape plot to {save_path}")

    return fig


def plot_minuit_summary(minuit_result: Dict[str, Any],
                       observed: Optional[np.ndarray] = None,
                       predicted: Optional[np.ndarray] = None,
                       x_coords: Optional[np.ndarray] = None,
                       y_coords: Optional[np.ndarray] = None,
                       title: str = "Minuit Optimization Summary",
                       figsize: Tuple[int, int] = (16, 12),
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive summary plot combining multiple visualizations.

    Args:
        minuit_result: Dictionary with Minuit optimization results
        observed: Optional observed data values
        predicted: Optional predicted data values
        x_coords: Optional x coordinates
        y_coords: Optional y coordinates
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    info = extract_minuit_info(minuit_result)
    params = info['parameters']
    n_params = len(params)

    # Create grid layout
    fig = plt.figure(figsize=figsize, dpi=DEFAULT_DPI)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Plot parameter values
    ax1 = fig.add_subplot(gs[0, :])
    param_names = list(params.keys())
    param_values = list(params.values())
    colors = [COLOR_BEST_FIT if name != 'fcn' else COLOR_INITIAL for name in param_names]
    bars = ax1.bar(param_names, param_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Parameter Value')
    ax1.set_title('Optimized Parameters')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, param_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=8)

    # Plot fit quality if data provided
    if observed is not None and predicted is not None:
        residuals = observed - predicted
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = 1 - np.sum(residuals**2) / np.sum((observed - np.mean(observed))**2)

        # Observed vs Predicted
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(observed, predicted, alpha=SCATTER_ALPHA, s=SCATTER_SIZE)
        min_val = min(observed.min(), predicted.min())
        max_val = max(observed.max(), predicted.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax2.set_xlabel('Observed')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Fit Quality')
        ax2.grid(True, alpha=0.3)

        # Residuals histogram
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(residuals, bins=HISTOGRAM_BINS, edgecolor='black', alpha=0.7)
        ax3.axvline(0, color='r', linewidth=2, linestyle='--')
        ax3.set_xlabel('Residual')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residual Distribution')
        ax3.text(0.98, 0.97, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}',
                transform=ax3.transAxes, verticalalignment='top',
                horizontalalignment='right', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.grid(True, alpha=0.3)

        # Residuals vs Index
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.scatter(range(len(residuals)), residuals, alpha=SCATTER_ALPHA, s=SCATTER_SIZE)
        ax4.axhline(0, color='r', linewidth=2, linestyle='--')
        ax4.set_xlabel('Index')
        ax4.set_ylabel('Residual')
        ax4.set_title('Residuals vs Index')
        ax4.grid(True, alpha=0.3)
    else:
        # Hide subplots if no data
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            ax.text(0.5, 0.5, 'No data provided', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

    # Statistics summary
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')

    summary_text = f"""
    OPTIMIZATION SUMMARY

    Function Value: {info['fcn_value']:.6f}
    Number of Parameters: {n_params}
    """

    if observed is not None and predicted is not None:
        residuals = observed - predicted
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        r2 = 1 - np.sum(residuals**2) / np.sum((observed - np.mean(observed))**2)
        summary_text += f"""

    FIT STATISTICS:
    RMSE: {rmse:.6f}
    MAE: {mae:.6f}
    R²: {r2:.6f}
    Data Points: {len(observed)}
    """

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info(f"Saved summary plot to {save_path}")

    return fig


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def visualize_minuit_optimization(minuit_result: Dict[str, Any],
                                 output_dir: str = '.',
                                 observed: Optional[np.ndarray] = None,
                                 predicted: Optional[np.ndarray] = None,
                                 x_coords: Optional[np.ndarray] = None,
                                 y_coords: Optional[np.ndarray] = None) -> Dict[str, str]:
    """
    Create all visualization plots and save them to disk.

    Args:
        minuit_result: Dictionary with Minuit optimization results
        output_dir: Directory to save plots
        observed: Optional observed data values
        predicted: Optional predicted data values
        x_coords: Optional x coordinates
        y_coords: Optional y coordinates

    Returns:
        Dictionary {plot_name: file_path} for all created plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    plots = {}

    # Parameter distributions
    info = extract_minuit_info(minuit_result)
    try:
        fig = plot_parameter_distributions(
            info['parameters'],
            title="Parameter Distributions",
            save_path=f"{output_dir}/minuit_parameters.png"
        )
        plots['parameters'] = f"{output_dir}/minuit_parameters.png"
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating parameter distribution plot: {e}")

    # Parameter correlations
    if len(info['parameters']) > 1:
        try:
            fig = plot_parameter_correlations(
                info['parameters'],
                title="Parameter Correlations",
                save_path=f"{output_dir}/minuit_correlations.png"
            )
            plots['correlations'] = f"{output_dir}/minuit_correlations.png"
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating correlation plot: {e}")

    # Residual analysis
    if observed is not None and predicted is not None:
        try:
            fig = plot_fit_residuals(
                observed, predicted, x_coords, y_coords,
                save_path=f"{output_dir}/minuit_residuals.png"
            )
            plots['residuals'] = f"{output_dir}/minuit_residuals.png"
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating residual plot: {e}")

    # Summary plot
    try:
        fig = plot_minuit_summary(
            minuit_result, observed, predicted, x_coords, y_coords,
            save_path=f"{output_dir}/minuit_summary.png"
        )
        plots['summary'] = f"{output_dir}/minuit_summary.png"
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating summary plot: {e}")

    logger.info(f"Created {len(plots)} visualization plots in {output_dir}")
    return plots