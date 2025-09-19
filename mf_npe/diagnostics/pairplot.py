# mf_npe/diagnostics/pairplot.py

"""
Provides a function to generate and save a pairplot of posterior samples,
overlaying the ground-truth parameter values for visual diagnosis.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from torch import Tensor


def plot_posterior_pairplot(
    posterior: NeuralPosterior,
    true_theta: Tensor,
    observation: Tensor,
    output_path: Path,
    num_samples: int = 2000,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Draws a pairplot of posterior samples and overlays the true parameters.

    Args:
        posterior: The trained sbi posterior distribution.
        true_theta: The ground-truth parameter vector, shape (dim_theta,).
        observation: The observation `x` to condition the posterior on.
        output_path: The full path (including filename) to save the plot PNG.
        num_samples: The number of samples to draw from the posterior for plotting.
        labels: A list of strings to use as labels for the parameter dimensions.
        title: An optional title for the entire plot.
    """
    # Generate samples from the posterior conditioned on the observation
    samples = posterior.sample((num_samples,), x=observation).detach().cpu().numpy()
    true_theta_np = true_theta.detach().cpu().numpy()
    dim_theta = samples.shape[1]

    if labels is None:
        labels = [f"θ$_{i+1}$" for i in range(dim_theta)]
    if len(labels) != dim_theta:
        raise ValueError(
            f"Number of labels ({len(labels)}) must match "
            f"parameter dimension ({dim_theta})."
        )

    df_samples = pd.DataFrame(samples, columns=labels)

    # Use the more flexible PairGrid to customize the plot
    grid = sns.PairGrid(df_samples)

    # Plot histograms on the diagonal
    grid.map_diag(sns.histplot, bins=30, kde=True)

    # Plot scatter plots on the off-diagonal
    grid.map_offdiag(sns.scatterplot, s=10, alpha=0.5, color="blue")

    # Overlay the ground-truth parameters
    for i in range(dim_theta):
        # Add a vertical line on the diagonal histograms
        grid.axes[i, i].axvline(true_theta_np[i], color='red', linestyle='--', lw=2)
        for j in range(dim_theta):
            if i != j:
                # Add a marker on the off-diagonal scatter plots
                grid.axes[i, j].plot(true_theta_np[j], true_theta_np[i], 'rx', markersize=10, mew=2)

    if title:
        plt.suptitle(title, y=1.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()