# mf_npe/simulator/simulation_func.py

"""
Provides a wrapper to connect the Hodgkin-Huxley simulators with the sbi package.

The SimulationWrapper class is the main entry point. It is initialized with all simulation parameters 
and can then be called like a function, making it directly compatible with sbi's `simulate_for_sbi`.
"""

from typing import Dict, Any

import jax.numpy as jnp
import numpy as np
import torch
from jax import random as jax_random
from torch import Tensor

from .low_fidelity_hh import NoisyHHSimulator
from .high_fidelity_hh import MarkovHHSimulator
from ..utils.utils import summarize_voltage


class SimulationWrapper:
    """
    A callable wrapper for running HH simulations for use with `sbi`.
    """
    def __init__(self, config: Dict[str, Any], key: jax_random.PRNGKey):
        """
        Initializes the wrapper with all fixed simulation parameters.

        Args:
            config: A dictionary containing all simulation parameters, e.g.,
                dt, t_max, stimulus info, channel counts, noise sigma, etc.
            key: A master JAX random number generator key.
        """
        self.config = config
        self.key = key

        sim_params = self.config.get("simulation", {})
        dt = sim_params.get("dt", 0.01)
        t_max = sim_params.get("t_max", 50.0)
        self.t_array_np = np.arange(0.0, t_max + dt, dt)
        self.t_array_jnp = jnp.array(self.t_array_np)

        hf_params = self.config.get("hf_specific", {})
        self.hf_simulator = MarkovHHSimulator(
            n_na_channels=hf_params.get("NNa", 6000),
            n_k_channels=hf_params.get("NK", 1800)
        )
        
        lf_params = self.config.get("lf_specific", {})
        self.lf_simulator = NoisyHHSimulator(
            sigma=lf_params.get("sigma", 2.0)
        )
        # -------------------------------------------

    def _run_lf(self, theta: Tensor) -> np.ndarray:
        """Runs the low-fidelity JAX-based simulator for a given theta."""
        self.key, subkey = jax_random.split(self.key)
        gNa, gK, gL = theta.tolist()
        
        self.lf_simulator.g_na, self.lf_simulator.g_k, self.lf_simulator.g_l = gNa, gK, gL
        
        stim_params = self.config.get("stimulus", {})
        _, v_trace = self.lf_simulator.simulate(
            key=subkey,
            t_array=self.t_array_jnp,
            i_amp=stim_params.get("amp", 10.0),
            i_delay=stim_params.get("delay", 10.0),
            i_dur=stim_params.get("dur", 20.0),
        )
        return np.array(v_trace)

    def _run_hf(self, theta: Tensor) -> np.ndarray:
        """Runs the high-fidelity NumPy-based simulator for a given theta."""
        gNa, gK, gL = theta.tolist()

        self.hf_simulator.g_na, self.hf_simulator.g_k, self.hf_simulator.g_l = gNa, gK, gL
        hf_params = self.config.get("hf_specific", {})
        base_NNa = hf_params.get("NNa", 6000)
        base_NK = hf_params.get("NK", 1800)
        self.hf_simulator.n_na = int(base_NNa * gNa / 120.0)
        self.hf_simulator.n_k = int(base_NK * gK / 36.0)

        stim_params = self.config.get("stimulus", {})
        def stimulus_func(time: float) -> float:
            delay = stim_params.get("delay", 10.0)
            dur = stim_params.get("dur", 20.0)
            amp = stim_params.get("amp", 10.0)
            return amp if delay <= time < (delay + dur) else 0.0

        sim_output = self.hf_simulator.simulate(
            t_array=self.t_array_np,
            stimulus_current_func=stimulus_func,
        )
        return sim_output[:, 1]  # voltage column

    def __call__(self, theta: Tensor, **kwargs) -> np.ndarray:
        """
        The main simulation entry point for `sbi`.

        This method runs the default simulator specified by the `fidelity`
        key in its configuration and returns the summary statistics.
        """
        fidelity = self.config.get("fidelity", "lf").lower()

        if fidelity.startswith("hf"):
            voltage_trace = self._run_hf(theta)
        else:
            voltage_trace = self._run_lf(theta)
        
        stim_config = self.config.get("stimulus", {})
        stim_start = stim_config.get("delay", 10.0)
        stim_end = stim_start + stim_config.get("dur", 20.0)

        return summarize_voltage(self.t_array_np, voltage_trace, stim_start, stim_end)
