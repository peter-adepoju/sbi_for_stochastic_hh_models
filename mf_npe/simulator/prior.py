# mf_npe/simulator/prior.py

"""
Defines the prior distribution for the Hodgkin-Huxley model parameters
and provides related utility functions.
"""

from typing import Dict, List, Tuple

import torch
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior
from torch import Tensor
from torch.distributions import Distribution

# Define a single source of truth for the parameter order.
PARAMETER_ORDER: Tuple[str, ...] = ("g_Na", "g_K", "g_Leak")


def create_prior(param_ranges: Dict[str, List[float]]) -> Distribution:
    """
    Creates a BoxUniform prior over the specified model parameters.

    Args:
        param_ranges: A dictionary mapping parameter names to their [min, max]
            range. Must include keys defined in `PARAMETER_ORDER`.

    Returns:
        An sbi-compatible prior distribution object.
    """
    try:
        low = torch.tensor([param_ranges[k][0] for k in PARAMETER_ORDER])
        high = torch.tensor([param_ranges[k][1] for k in PARAMETER_ORDER])
    except KeyError as e:
        raise ValueError(
            f"param_ranges dictionary is missing a required key: {e}. "
            f"It must contain: {PARAMETER_ORDER}"
        )

    prior = BoxUniform(low=low.float(), high=high.float())

    # `process_prior` ensures the prior is compatible with the sbi backend.
    processed_prior, _, _ = process_prior(prior)

    return processed_prior


def filter_invalid_summaries(summary_stats: Tensor) -> Tensor:
    """
    Creates a boolean mask to filter out invalid simulation summaries.

    Invalid summaries are those containing NaN or Inf values.

    Args:
        summary_stats: A tensor of summary statistics, with shape
            (num_simulations, num_stats).

    Returns:
        A boolean tensor of shape (num_simulations,) where `True` indicates a
        valid summary statistic.
    """
    is_nan = torch.isnan(summary_stats).any(dim=1)
    is_inf = torch.isinf(summary_stats).any(dim=1)
    valid_mask = ~is_nan & ~is_inf
    return valid_mask


def get_plotting_ranges(param_ranges: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    Formats parameter ranges into a dictionary suitable for plotting libraries.

    Converts a dictionary like `{'g_Na': [min, max], ...}` into
    `{'range_theta1': [min, max], ...}`.

    Args:
        param_ranges: A dictionary mapping parameter names to their [min, max]
            range.

    Returns:
        A dictionary with keys formatted for plotting.
    """
    return {
        f"range_theta{i+1}": param_ranges[k]
        for i, k in enumerate(PARAMETER_ORDER)
        if k in param_ranges
    }