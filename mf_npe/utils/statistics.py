# mf_npe/utils/statistics.py

"""
Provides utility functions for calculating summary statistics.
"""

from typing import Tuple

import torch
from torch import Tensor


def calculate_mean_and_ci(
    values: Tensor,
    ci_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Computes the mean and confidence interval bounds for a tensor of values.

    Args:
        values: A 1D tensor of numerical values.
        ci_level: The desired confidence interval level (e.g., 0.95 for 95% CI).

    Returns:
        A tuple containing (mean, lower_bound, upper_bound).
        Returns (NaN, NaN, NaN) if the input tensor is empty.
    """
    if values.numel() == 0:
        return float('nan'), float('nan'), float('nan')

    mean = torch.mean(values).item()

    # Calculate quantiles for a two-sided interval
    alpha = (1.0 - ci_level) / 2.0
    lower_bound = torch.quantile(values, alpha).item()
    upper_bound = torch.quantile(values, 1.0 - alpha).item()

    return mean, lower_bound, upper_bound