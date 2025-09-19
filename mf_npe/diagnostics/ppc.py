# mf-npe/diagnostics/ppc.py

"""
Performs and plots posterior-predictive checks (PPC).

This module provides a function to generate simulations from a trained posterior, calculate summary statistics 
from those simulations, and plot their distributions against the observed summary statistics.
"""

from pathlib import Path
from typing import Callable, List, Dict, Any

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from torch import Tensor

from mf_npe.utils.utils import summarize_voltage


def _plot_ppc_histograms(
    sim_stats: np.ndarray,
    obs_stats: np.ndarray,
    labels: List[str],
    title: str,
    save_path: Path,
    plotting_config: Dict[str, Any],
) -> None:
    """
    Creates and saves a grid of histograms comparing simulated and observed stats.

    Args:
        sim_stats: Array of summary statistics from posterior simulations.
        obs_stats: Array of summary statistics from the observed data.
        labels: Names of the summary statistics for subplot titles.
        title: The main title for the figure.
        save_path: The full path to save the HTML figure.
        plotting_config: A dictionary with style settings (e.g., width, height).
    """
    n_stats = sim_stats.shape[1]
    cols = min(n_stats, 4)
    rows = (n_stats - 1) // cols + 1

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=labels)

    for i in range(n_stats):
        r, c = divmod(i, cols)
        fig.add_trace(
            go.Histogram(x=sim_stats[:, i], opacity=0.75, name=labels[i]),
            row=r + 1, col=c + 1,)
        # Add a vertical line for the observed statistic
        fig.add_vline(
            x=obs_stats[i], line_width=3, line_dash="dash", line_color="red",
            row=r + 1, col=c + 1)

    fig.update_layout(
        title=title,
        width=plotting_config.get("width_px", 800),
        height=plotting_config.get("height_px", 600),
        bargap=0.05,
        showlegend=False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))

    if plotting_config.get("show_plots_default", False):
        fig.show()


def run_and_plot_ppc(
    posterior: NeuralPosterior,
    observation: Tensor,
    time_points: np.ndarray,
    observed_trace: np.ndarray,
    simulator: Callable[[Tensor], np.ndarray],
    output_path: Path,
    title: str,
    num_ppc_samples: int = 100,
    summary_labels: List[str] = ["Spike Count", "Mean Rest", "Std Rest", "Mean Stim"],
    plotting_config: Dict[str, Any] = {},
) -> None:
    """
    Runs a posterior-predictive check and saves the resulting plot.

    Args:
        posterior: The trained sbi posterior.
        observation: The summary statistic of the observed data, used to condition the posterior.
        time_points: The time array for the simulation.
        observed_trace: The raw voltage trace of the observed data.
        simulator: A callable that takes a parameter tensor and returns its summary statistic.
        output_path: The full path (including filename) to save the plot.
        title: The title for the plot.
        num_ppc_samples: The number of samples to draw from the posterior.
        summary_labels: Labels for the summary statistics.
        plotting_config: A dictionary with plot style settings.
    """
    # Draw samples from the posterior
    posterior_samples = posterior.sample((num_ppc_samples,), x=observation)

    # For each sample, run a simulation and get its summary stats
    simulated_summaries = []
    for theta in posterior_samples:
        summary = simulator(theta)
        simulated_summaries.append(summary)

    simulated_summaries = np.stack(simulated_summaries)

    observed_summary = summarize_voltage(time_points, observed_trace)

    print(f"Generating PPC plot and saving to: {output_path}")
    _plot_ppc_histograms(
        sim_stats=simulated_summaries,
        obs_stats=observed_summary,
        labels=summary_labels,
        title=title,
        save_path=output_path,
        plotting_config=plotting_config)
