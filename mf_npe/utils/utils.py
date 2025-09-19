# mf_npe/utils/utils.py

"""
A collection of general-purpose utility functions for the project.
"""

import pickle
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> Optional["jax.random.PRNGKey"]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import jax
        from jax import random as jax_random
        key = jax_random.PRNGKey(seed)
        return key
    except ImportError:
        return None


def summarize_voltage(
    t: np.ndarray,
    v: np.ndarray,
    stim_start: float = 10.0,
    stim_end: float = 30.0
) -> np.ndarray:
    """
    Computes a 4-dimensional summary statistic vector from a voltage trace.

    The summary statistics are:
    1.  num_spikes: Count of upward crossings of 0 mV.
    2.  mean_rest: Mean voltage before the stimulus.
    3.  std_rest: Standard deviation of voltage before the stimulus.
    4.  mean_stim: Mean voltage during the stimulus.

    Args:
        t: The time array for the voltage trace.
        v: The voltage trace array.
        stim_start: The start time of the stimulus window.
        stim_end: The end time of the stimulus window.

    Returns:
        A 4-element NumPy array of the summary statistics.
    """
    # Count spikes
    is_spike = (v[:-1] < 0) & (v[1:] >= 0)
    num_spikes = np.sum(is_spike)

    # Resting statistics (pre-stimulus)
    rest_mask = t < stim_start
    v_rest = v[rest_mask]
    mean_rest = np.mean(v_rest) if v_rest.size > 0 else np.nan
    std_rest = np.std(v_rest) if v_rest.size > 0 else np.nan

    # Stimulus statistics
    stim_mask = (t >= stim_start) & (t < stim_end)
    v_stim = v[stim_mask]
    mean_stim = np.mean(v_stim) if v_stim.size > 0 else np.nan

    return np.array([num_spikes, mean_rest, std_rest, mean_stim], dtype=np.float32)


def dump_pickle(path: Path, variable: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(variable, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved variable to {path}")


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        variable = pickle.load(f)
    print(f"Loaded variable from {path}")
    return variable