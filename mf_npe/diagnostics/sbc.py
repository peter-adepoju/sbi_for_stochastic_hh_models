# mf_npe/diagnostics/sbc.py

"""
Performs Simulation-Based Calibration (SBC) to check posterior calibration.

This module provides functions to:
1. Run SBC: Generate calibration data, sample from one or more posteriors,
   and compute the rank statistics.
2. Plot SBC results: Create histogram and CDF plots of the rank statistics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from sbi.analysis import sbc_rank_plot
from sbi.diagnostics.sbc import run_sbc as sbi_run_sbc
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from torch import Tensor
from torch.distributions import Distribution

from mf_npe.simulator.simulation_func import SimulationWrapper


def run_sbc(
    posteriors: Dict[str, NeuralPosterior],
    prior: Distribution,
    simulator: SimulationWrapper,
    num_sbc_runs: int,
    num_posterior_draws: int,
) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
    """
    Runs SBC for one or more posteriors and returns the rank statistics.

    Args:
        posteriors: A dictionary mapping a name (e.g., "HF") to a trained posterior.
        prior: The prior distribution used to generate calibration data.
        simulator: A configured SimulationWrapper to generate calibration data.
        num_sbc_runs: The number of simulations to generate for the SBC check.
        num_posterior_draws: The number of samples to draw from the posterior
            for each simulation.

    Returns:
        A tuple containing:
        - A dictionary mapping posterior names to their computed rank statistics.
        - The ground-truth parameters (`thetas`) used for calibration.
        - The summary statistics (`xs`) generated for calibration.
    """
    print(f"Generating {num_sbc_runs} simulations for SBC...")
    # Generate ground-truth parameters from the prior
    thetas = prior.sample((num_sbc_runs,))
    # Generate corresponding observations (summary statistics) using the simulator
    xs = torch.stack([torch.from_numpy(simulator(theta)) for theta in thetas])

    ranks_dict = {}
    for name, posterior in posteriors.items():
        print(f"Running SBC for '{name}' posterior...")
        # sbi's run_sbc handles the posterior sampling and rank calculation efficiently
        ranks = sbi_run_sbc(
            theta=thetas,
            x=xs,
            posterior=posterior,
            num_posterior_samples=num_posterior_draws)
        ranks_dict[name] = ranks

    return ranks_dict, thetas, xs


def plot_sbc_ranks(
    ranks_dict: Dict[str, Tensor],
    num_posterior_samples: int,
    output_path: Path,
    parameter_labels: List[str],
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Generates and saves histogram and CDF plots for SBC ranks.

    Args:
        ranks_dict: A dictionary mapping a name to a tensor of rank statistics.
        num_posterior_samples: The number of posterior draws used to compute ranks.
        output_path: The base path for saving the plot files (without extension).
        parameter_labels: Names of the parameters for plot titles.
        colors: An optional dictionary mapping names to plot colors.
    """
    if colors is None:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = {name: default_colors[i % len(default_colors)]
                  for i, name in enumerate(ranks_dict.keys())}

    # --- Create Histogram Plot ---
    fig_hist, axes_hist = plt.subplots(1, len(parameter_labels), figsize=(4 * len(parameter_labels), 4))
    for name, ranks in ranks_dict.items():
        sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="hist",
            parameter_names=parameter_labels,
            fig=fig_hist,
            axes=axes_hist,
            legend_inside=True,
            hist_bar_color=colors[name],
            label=name)
    fig_hist.tight_layout()
    fig_hist.savefig(output_path.with_suffix(".hist.svg"))
    plt.close(fig_hist)
    print(f"Saved SBC histogram to {output_path.with_suffix('.hist.svg')}")

    # --- Create CDF Plot ---
    fig_cdf, axes_cdf = plt.subplots(1, 1, figsize=(6, 5))
    for name, ranks in ranks_dict.items():
        sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="cdf",
            fig=fig_cdf,
            axes=axes_cdf,
            cdf_line_color=colors[name],
            label=name)
    axes_cdf.legend()
    fig_cdf.tight_layout()
    fig_cdf.savefig(output_path.with_suffix(".cdf.svg"))
    plt.close(fig_cdf)
    print(f"Saved SBC CDF plot to {output_path.with_suffix('.cdf.svg')}")