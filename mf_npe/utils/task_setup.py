# mf_npe/utils/task_setup.py

"""
Provides a factory for loading task-specific configurations.

This module defines all the constants and simulator functions for different
scientific models in the project. The main entry point is `load_task_config`.
"""

from typing import Dict, Any, Tuple, Callable

from mf_npe.simulator.low_fidelity_hh import hh_current_noise
from mf_npe.simulator.high_fidelity_hh import markov_hh


def _get_hh_model_config() -> Tuple[Dict[str, Any], Callable, Callable]:
    """
    Constructs the configuration dictionary and simulators for the HH model.
    """
    config_data = {
        # --- Prior Configuration ---
        "prior_ranges": {
            "g_Na":   [50.0, 200.0],
            "g_K":    [20.0, 100.0],
            "g_Leak": [0.1,   1.0],
        },
        "theta_dim": 3,  # We infer g_Na, g_K, g_Leak

        # --- Summary Statistic Configuration ---
        "x_dim": 4,      # num_spikes, mean_rest, std_rest, mean_stim

        # --- Fidelity Labels ---
        "type_lf": "lf",
        "type_hf": "hf",

        # --- Simulation Parameters ---
        "simulation": {
            "dt": 0.01,   # ms
            "t_max": 50.0,   # ms
        },

        # --- Stimulus Parameters ---
        "stimulus": {
            "delay": 10.0,  # ms
            "dur":   20.0,  # ms
            "amp":   10.0,  # μA/cm²
        },
        
        # --- Fidelity-Specific Parameters ---
        "lf_specific": {
            "sigma": 2.0,   # Current-noise standard deviation
        },
        "hf_specific": {
            "NNa": 6000,    # Number of sodium channels
            "NK": 1800,     # Number of potassium channels
        },
    }

    # Assign the simulator functions
    lf_simulator = hh_current_noise
    hf_simulator = markov_hh

    return config_data, lf_simulator, hf_simulator


# A registry to map task names to their configuration-loading functions. This makes it easy to add new models in the future.
_TASK_REGISTRY = {"hh_model": _get_hh_model_config}


def load_task_config(sim_name: str) -> Tuple[Dict[str, Any], Callable, Callable]:
    """
    Loads the configuration data and simulator functions for a given task name.

    Args:
        sim_name: The name of the simulation task to load.

    Returns:
        A tuple containing:
        - A dictionary of all task-specific constants and parameters.
        - The low-fidelity simulator function.
        - The high-fidelity simulator function.

    Raises:
        NotImplementedError: If the `sim_name` is not found in the task registry.
    """
    if sim_name not in _TASK_REGISTRY:
        raise NotImplementedError(
            f"Simulation task '{sim_name}' is not recognized. "
            f"Available tasks are: {list(_TASK_REGISTRY.keys())}"
        )
    
    return _TASK_REGISTRY[sim_name]()