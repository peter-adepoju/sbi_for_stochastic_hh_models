# mf_npe/simulator/high_fidelity_hh.py

"""
High-fidelity Hodgkin-Huxley neuron model with stochastic ion channel kinetics.

This simulator implements the 8-state sodium channel and 5-state potassium channel Markov models, 
using a Gillespie stochastic simulation algorithm (SSA) to simulate individual channel transitions.

The model implementation is adapted from:
Goldwyn, J. H., & Shea-Brown, E. (2011). The what and where of adding channel noise 
to the Hodgkin-Huxley equations. PLoS computational biology, 7(11), e1002247.
"""

from typing import Callable, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Numerically-stable rate functions and helpers
# -----------------------------------------------------------------------------

def _efun(z: np.ndarray) -> np.ndarray:
    """Numerically stable version of z / (exp(z) - 1)."""
    # Clip to avoid overflow in np.exp
    z = np.clip(z, -500, 500)
    return np.where(
        np.abs(z) < 1e-6,
        1 - z / 2,          # Taylor expansion for small z
        z / (np.exp(z) - 1)
    )

# --- Rate functions for gating variables (m, h, n) ---
# Defined on relative voltage V_rel = V - V_rest

def _alpha_m(V_rel: float) -> float:
    return 10.0 * _efun((V_rel - 25.0) / -10.0)

def _beta_m(V_rel: float) -> float:
    return 4.0 * np.exp(V_rel / -18.0)

def _alpha_h(V_rel: float) -> float:
    return 0.07 * np.exp(V_rel / -20.0)

def _beta_h(V_rel: float) -> float:
    return 1.0 / (np.exp((V_rel - 30.0) / 10.0) + 1.0)

def _alpha_n(V_rel: float) -> float:
    return 1.0 * _efun((V_rel - 10.0) / -10.0)

def _beta_n(V_rel: float) -> float:
    return 0.125 * np.exp(V_rel / -80.0)


class MarkovHHSimulator:
    """
    High-fidelity Hodgkin-Huxley simulator with Markov chain channel kinetics.
    """
    # This static list defines the 28 possible state transitions for the channels.
    _STATE_CHANGES = [
        # Na_m open (0-2)
        ((0, (0, 0), -1), (0, (1, 0), 1)), ((0, (1, 0), -1), (0, (2, 0), 1)), ((0, (2, 0), -1), (0, (3, 0), 1)),
        # Na_m close (3-5)
        ((0, (3, 0), -1), (0, (2, 0), 1)), ((0, (2, 0), -1), (0, (1, 0), 1)), ((0, (1, 0), -1), (0, (0, 0), 1)),
        # Na_h inactivate (6-9)
        ((0, (0, 0), -1), (0, (0, 1), 1)), ((0, (1, 0), -1), (0, (1, 1), 1)), ((0, (2, 0), -1), (0, (2, 1), 1)), ((0, (3, 0), -1), (0, (3, 1), 1)),
        # Na_h recover (10-13)
        ((0, (0, 1), -1), (0, (0, 0), 1)), ((0, (1, 1), -1), (0, (1, 0), 1)), ((0, (2, 1), -1), (0, (2, 0), 1)), ((0, (3, 1), -1), (0, (3, 0), 1)),
        # Inactivated Na_m open (14-16)
        ((0, (0, 1), -1), (0, (1, 1), 1)), ((0, (1, 1), -1), (0, (2, 1), 1)), ((0, (2, 1), -1), (0, (3, 1), 1)),
        # Inactivated Na_m close (17-19)
        ((0, (3, 1), -1), (0, (2, 1), 1)), ((0, (2, 1), -1), (0, (1, 1), 1)), ((0, (1, 1), -1), (0, (0, 1), 1)),
        # K_n open (20-23)
        ((1, 0, -1), (1, 1, 1)), ((1, 1, -1), (1, 2, 1)), ((1, 2, -1), (1, 3, 1)), ((1, 3, -1), (1, 4, 1)),
        # K_n close (24-27)
        ((1, 4, -1), (1, 3, 1)), ((1, 3, -1), (1, 2, 1)), ((1, 2, -1), (1, 1, 1)), ((1, 1, -1), (1, 0, 1)),
    ]

    def __init__(self,
                 g_na: float = 120.0, g_k: float = 36.0, g_l: float = 0.3,
                 e_na: float = 50.0, e_k: float = -77.0, e_l: float = -54.4,
                 c_m: float = 1.0, v_rest: float = -65.0,
                 n_na_channels: int = 6000, n_k_channels: int = 1800):
        """
        Initializes the simulator with model parameters.

        Args:
            g_na, g_k, g_l: Max conductances (mS/cm²).
            e_na, e_k, e_l: Reversal potentials (mV). These are absolute, not relative.
            c_m: Membrane capacitance (µF/cm²).
            v_rest: Resting membrane potential (mV).
            n_na_channels, n_k_channels: Total number of channels.
        """
        self.g_na, self.g_k, self.g_l = g_na, g_k, g_l
        self.e_na, self.e_k, self.e_l = e_na, e_k, e_l
        self.c_m = c_m
        self.v_rest = v_rest
        self.n_na = n_na_channels
        self.n_k = n_k_channels

        # Initial steady-state gating values at V_rel = 0
        m0 = _alpha_m(0.0) / (_alpha_m(0.0) + _beta_m(0.0))
        h0 = _alpha_h(0.0) / (_alpha_h(0.0) + _beta_h(0.0))
        n0 = _alpha_n(0.0) / (_alpha_n(0.0) + _beta_n(0.0))

        # Initialize Markov state populations based on steady-state probabilities
        self.na_channel_states = np.zeros((4, 2))  # (m_state, h_state)
        self.na_channel_states[0, 0] = np.round(self.n_na * (1-m0)**3 * (1-h0))
        self.na_channel_states[1, 0] = np.round(self.n_na * 3*m0*(1-m0)**2 * (1-h0))
        self.na_channel_states[2, 0] = np.round(self.n_na * 3*m0**2*(1-m0) * (1-h0))
        self.na_channel_states[3, 0] = np.round(self.n_na * m0**3 * (1-h0))
        self.na_channel_states[0, 1] = np.round(self.n_na * (1-m0)**3 * h0)
        self.na_channel_states[1, 1] = np.round(self.n_na * 3*m0*(1-m0)**2 * h0)
        self.na_channel_states[2, 1] = np.round(self.n_na * 3*m0**2*(1-m0) * h0)
        self.na_channel_states[3, 1] = np.round(self.n_na * m0**3 * h0)
        # Ensure total is correct due to rounding
        self.na_channel_states[3, 1] += self.n_na - np.sum(self.na_channel_states)

        self.k_channel_states = np.zeros(5) # (n_state)
        self.k_channel_states[0] = np.round(self.n_k * (1-n0)**4)
        self.k_channel_states[1] = np.round(self.n_k * 4*n0*(1-n0)**3)
        self.k_channel_states[2] = np.round(self.n_k * 6*n0**2*(1-n0)**2)
        self.k_channel_states[3] = np.round(self.n_k * 4*n0**3*(1-n0))
        self.k_channel_states[4] = np.round(self.n_k * n0**4)
        # Ensure total is correct
        self.k_channel_states[4] += self.n_k - np.sum(self.k_channel_states)

    def _gillespie_step(self, v_membrane: float, t_start: float, dt: float) -> None:
        """
        Performs one Gillespie SSA step to update channel states over dt.
        """
        t_switch = t_start
        v_rel = v_membrane - self.v_rest

        am, bm = _alpha_m(v_rel), _beta_m(v_rel)
        ah, bh = _alpha_h(v_rel), _beta_h(v_rel)
        an, bn = _alpha_n(v_rel), _beta_n(v_rel)
        
        # State arrays for easier access
        na_s = self.na_channel_states
        k_s = self.k_channel_states
        
        while t_switch < (t_start + dt):
            propensities = np.array([
                3*am*na_s[0,0], 2*am*na_s[1,0], 1*am*na_s[2,0],
                3*bm*na_s[3,0], 2*bm*na_s[2,0], 1*bm*na_s[1,0],
                ah*na_s[0,0], ah*na_s[1,0], ah*na_s[2,0], ah*na_s[3,0],
                bh*na_s[0,1], bh*na_s[1,1], bh*na_s[2,1], bh*na_s[3,1],
                3*am*na_s[0,1], 2*am*na_s[1,1], 1*am*na_s[2,1],
                3*bm*na_s[3,1], 2*bm*na_s[2,1], 1*bm*na_s[1,1],
                4*an*k_s[0], 3*an*k_s[1], 2*an*k_s[2], 1*an*k_s[3],
                4*bn*k_s[4], 3*bn*k_s[3], 2*bn*k_s[2], 1*bn*k_s[1]
            ])

            total_rate = np.sum(propensities)
            if total_rate == 0:
                break

            # Draw waiting time from an exponential distribution
            t_update = -np.log(np.random.rand()) / total_rate
            t_switch += t_update
            if t_switch >= (t_start + dt):
                break

            # Determine which transition occurs
            r = total_rate * np.random.rand()
            cumulative_rates = np.cumsum(propensities)
            reaction_index = np.searchsorted(cumulative_rates, r, side='right')
            
            # Apply the state change
            state_updates = self._STATE_CHANGES[reaction_index]
            channel_arrays = [self.na_channel_states, self.k_channel_states]
            for array_idx, state_idx, change in state_updates:
                channel_arrays[array_idx][state_idx] += change

    def simulate(self,
                 t_array: np.ndarray,
                 stimulus_current_func: Callable[[float], float]
                 ) -> np.ndarray:
        """
        Runs the simulation.

        Args:
            t_array: Array of time points for the simulation (ms).
            stimulus_current_func: A function I(t) that returns the stimulus
                                   current (µA/cm²) at time t.

        Returns:
            A NumPy array containing the simulation output, with columns for
            [time, voltage, Na_open_fraction, K_open_fraction].
        """
        dt = t_array[1] - t_array[0]
        num_steps = len(t_array)
        
        v_membrane = self.v_rest
        
        # Preallocate output array
        output = np.zeros((num_steps, 4))
        output[0, 0] = t_array[0]
        output[0, 1] = v_membrane
        output[0, 2] = self.na_channel_states[3, 1] / self.n_na
        output[0, 3] = self.k_channel_states[4] / self.n_k

        for i in range(1, num_steps):
            t_prev = t_array[i - 1]

            # Update channel states using Gillespie algorithm
            self._gillespie_step(v_membrane, t_prev, dt)
            
            na_open_fraction = self.na_channel_states[3, 1] / self.n_na
            k_open_fraction = self.k_channel_states[4] / self.n_k
            
            # Update membrane potential using exponential Euler
            i_na = self.g_na * na_open_fraction * (v_membrane - self.e_na)
            i_k = self.g_k * k_open_fraction * (v_membrane - self.e_k)
            i_l = self.g_l * (v_membrane - self.e_l)
            i_stim = stimulus_current_func(t_prev)
            
            dv_dt = (i_stim - (i_na + i_k + i_l)) / self.c_m
            v_membrane += dt * dv_dt 
            
            # Results
            output[i, 0] = t_array[i]
            output[i, 1] = v_membrane
            output[i, 2] = na_open_fraction
            output[i, 3] = k_open_fraction
            
        return output