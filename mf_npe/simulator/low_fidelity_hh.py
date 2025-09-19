# mf_npe/simulator/low_fidelity_hh.py

"""
Low-fidelity Hodgkin-Huxley neuron model with additive current noise.

This simulator uses JAX for a high-performance, just-in-time (JIT) compiled implementation. 
Stochasticity is introduced as Gaussian noise added directly to the membrane potential update, 
providing a fast but approximate alternative to the high-fidelity Markov chain model.

The exponential-Euler integration method is adapted from:
Goldwyn, J. H., & Shea-Brown, E. (2011). The what and where of adding channel noise 
to the Hodgkin-Huxley equations. PLoS computational biology, 7(11), e1002247.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import jit, lax, random


# -----------------------------------------------------------------------------
# Module-level private helper functions
# -----------------------------------------------------------------------------

def _Exp(z: jnp.ndarray) -> jnp.ndarray:
    """Clips input to jnp.exp to avoid overflow."""
    return jnp.exp(jnp.clip(z, -500, None))

def _efun(x, y) -> jnp.ndarray:
    """Numerically stable version of z / (exp(z) - 1)."""
    return jnp.where(jnp.abs(x / y) < 1e-6, y * (1 - x / (2 * y)), x / (_Exp(x / y) - 1))

# --- Rate functions for gating variables (m, h, n) ---
# Defined on relative voltage V_rel = V - V_rest

def _alpha_m(V: float) -> float: return 0.1 * _efun(25.0 - V, 10.0)
def _beta_m(V: float) -> float:  return 4.0 * _Exp(-V / 18.0)
def _alpha_h(V: float) -> float: return 0.07 * _Exp(- V / 20.0)
def _beta_h(V: float) -> float:  return 1.0 / (_Exp((30.0 - V) / 10.0) + 1.0)
def _alpha_n(V: float) -> float: return 0.01 * _efun(10.0 - V, 10.0)
def _beta_n(V: float) -> float:  return 0.125 * _Exp(- V / 80.0)

def _tau_x(alpha_func, beta_func, V): return 1.0 / (alpha_func(V) + beta_func(V))
def _x_inf(alpha_func, beta_func, V): return alpha_func(V) * _tau_x(alpha_func, beta_func, V)


class NoisyHHSimulator:
    """
    Low-fidelity Hodgkin-Huxley simulator with additive current noise.
    """
    def __init__(self,
                 g_na: float = 120.0, g_k: float = 36.0, g_l: float = 0.3,
                 c_m: float = 1.0, v_rest: float = -65.0, sigma: float = 2.0):
        """
        Initializes the simulator with model parameters.

        Args:
            g_na, g_k, g_l: Max conductances (mS/cm²).
            c_m: Membrane capacitance (µF/cm²).
            v_rest: Resting membrane potential (mV).
            sigma: Amplitude of the additive Gaussian current noise.
        """
        self.g_na, self.g_k, self.g_l = g_na, g_k, g_l
        self.c_m = c_m
        self.v_rest = v_rest
        self.sigma = sigma

        # Reversal potentials relative to V_rest
        e_na_rel, e_k_rel, e_l_rel = 115.0, -12.0, 10.6
        self.e_na = e_na_rel + self.v_rest
        self.e_k = e_k_rel + self.v_rest
        self.e_l = e_l_rel + self.v_rest

    def simulate(self,
                 key: jax.random.PRNGKey,
                 t_array: jnp.ndarray,
                 i_amp: float = 10.0,
                 i_delay: float = 10.0,
                 i_dur: float = 20.0
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Runs a simulation with a step-current stimulus.

        Args:
            key: JAX random number generator key.
            t_array: Array of time points for the simulation (ms).
            i_amp: Amplitude of the stimulus current (µA/cm²).
            i_delay: Start time of the stimulus (ms).
            i_dur: Duration of the stimulus (ms).

        Returns:
            A tuple of (time_array, voltage_trace).
        """
        dt = t_array[1] - t_array[0]
        i_ext = jnp.where((t_array >= i_delay) & (t_array < i_delay + i_dur), i_amp, 0.0)

        # Define the single-step update function for lax.scan
        def _step_func(carry, inputs):
            v_prev, m_prev, h_prev, n_prev = carry
            i_t, noise_t = inputs
            v_rel_prev = v_prev - self.v_rest

            # Conductances
            g_na_val = self.g_na * m_prev**3 * h_prev
            g_k_val = self.g_k * n_prev**4
            g_total = g_na_val + g_k_val + self.g_l

            # Deterministic voltage update (exponential Euler)
            v_inf = (g_na_val * self.e_na + g_k_val * self.e_k + self.g_l * self.e_l + i_t) / g_total
            tau_v = self.c_m / g_total
            v_det = v_inf + (v_prev - v_inf) * _Exp(-dt / tau_v)

            # Additive noise term
            v_new = v_det + (self.sigma / self.c_m) * jnp.sqrt(dt) * noise_t

            # Gating variable updates (exponential Euler)
            v_rel = v_new - self.v_rest
            m_inf = _x_inf(_alpha_m, _beta_m, v_rel)
            h_inf = _x_inf(_alpha_h, _beta_h, v_rel)
            n_inf = _x_inf(_alpha_n, _beta_n, v_rel)
            m_new = m_inf + (m_prev - m_inf) * _Exp(-dt / _tau_x(_alpha_m, _beta_m, v_rel))
            h_new = h_inf + (h_prev - h_inf) * _Exp(-dt / _tau_x(_alpha_h, _beta_h, v_rel))
            n_new = n_inf + (n_prev - n_inf) * _Exp(-dt / _tau_x(_alpha_n, _beta_n, v_rel))

            new_carry = (v_new, m_new, h_new, n_new)
            output = v_new  
            return new_carry, output

        # Initial state at rest (V_rel = 0)
        v0 = self.v_rest
        m0 = _x_inf(_alpha_m, _beta_m, 0.0)
        h0 = _x_inf(_alpha_h, _beta_h, 0.0)
        n0 = _x_inf(_alpha_n, _beta_n, 0.0)
        initial_state = (v0, m0, h0, n0)

        # Generate noise sequence
        noise = random.normal(key, shape=t_array.shape)

        # Run simulation
        scan_jitted = jit(lax.scan, static_argnums=(0,))
        final_state, v_trace = scan_jitted(_step_func, initial_state, (i_ext, noise))

        return t_array, v_trace
