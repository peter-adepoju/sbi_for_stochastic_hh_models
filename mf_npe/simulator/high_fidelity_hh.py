# mf_npe/simulator/high_fidelity_hh.py

"""
High-fidelity Hodgkin-Huxley neuron model with stochastic ion channel kinetics.

This file provides a standalone function `markov_hh` that implements the
8-state sodium and 5-state potassium channel Markov models using a Gillespie
stochastic simulation algorithm (SSA).

The model implementation is adapted from:
Goldwyn, J. H., & Shea-Brown, E. (2011). The what and where of adding channel noise 
to the Hodgkin-Huxley equations. PLoS computational biology, 7(11), e1002247.
"""

import numpy as np
from typing import Callable

# -----------------------------------------------------------------------------
# Numerically‐stable helpers (clip inside exp to avoid under/overflow)
# -----------------------------------------------------------------------------
def _Exp(z):
    return np.exp(np.clip(z, -500, None))

def _efun(x, y):
    return np.where(
        np.abs(x / y) < 1e-6,
        y * (1 - x / (2 * y)),
        x / (_Exp(x / y) - 1)
    )

# -----------------------------------------------------------------------------
# Rate functions for gating variables (m, h, n)
# Defined on relative voltage V_rel = V - V_rest
# -----------------------------------------------------------------------------
def alpha_m(V): return 0.1 * _efun(25.0 - V, 10.0)
def beta_m(V):  return 4.0 * _Exp(-V / 18.0)
def alpha_h(V): return 0.07 * _Exp(-V / 20.0)
def beta_h(V):  return 1.0 / (_Exp((30.0 - V) / 10.0) + 1.0)
def alpha_n(V): return 0.01 * _efun(10.0 - V, 10.0)
def beta_n(V):  return 0.125 * _Exp(-V / 80.0)

def tau_x(a, b, V): return 1.0 / (a(V) + b(V))
def x_inf(a, b, V): return a(V) * tau_x(a, b, V)

# -----------------------------------------------------------------------------
# Markov chain Gillespie update function
# -----------------------------------------------------------------------------
def markov_chain_fraction(V, NaStateIn, KStateIn, t, dt, V_rest):
    Nastate = NaStateIn.copy()
    Kstate  = KStateIn.copy()
    tswitch = t

    while tswitch < (t + dt):
        V_rel = V - V_rest

        rate = np.zeros(28)
        rate[0]  = 3 * alpha_m(V_rel) * Nastate[0, 0]
        rate[1]  = rate[0]  + 2 * alpha_m(V_rel) * Nastate[1, 0]
        rate[2]  = rate[1]  + 1 * alpha_m(V_rel) * Nastate[2, 0]
        rate[3]  = rate[2]  + 3 * beta_m(V_rel)  * Nastate[3, 0]
        rate[4]  = rate[3]  + 2 * beta_m(V_rel)  * Nastate[2, 0]
        rate[5]  = rate[4]  + 1 * beta_m(V_rel)  * Nastate[1, 0]
        rate[6]  = rate[5]  +       alpha_h(V_rel) * Nastate[0, 0]
        rate[7]  = rate[6]  +       alpha_h(V_rel) * Nastate[1, 0]
        rate[8]  = rate[7]  +       alpha_h(V_rel) * Nastate[2, 0]
        rate[9]  = rate[8]  +       alpha_h(V_rel) * Nastate[3, 0]
        rate[10] = rate[9]  +       beta_h(V_rel)  * Nastate[0, 1]
        rate[11] = rate[10] +       beta_h(V_rel)  * Nastate[1, 1]
        rate[12] = rate[11] +       beta_h(V_rel)  * Nastate[2, 1]
        rate[13] = rate[12] +       beta_h(V_rel)  * Nastate[3, 1]
        rate[14] = rate[13] + 3 * alpha_m(V_rel) * Nastate[0, 1]
        rate[15] = rate[14] + 2 * alpha_m(V_rel) * Nastate[1, 1]
        rate[16] = rate[15] + 1 * alpha_m(V_rel) * Nastate[2, 1]
        rate[17] = rate[16] + 3 * beta_m(V_rel)  * Nastate[3, 1]
        rate[18] = rate[17] + 2 * beta_m(V_rel)  * Nastate[2, 1]
        rate[19] = rate[18] + 1 * beta_m(V_rel)  * Nastate[1, 1]
        rate[20] = rate[19] + 4 * alpha_n(V_rel) * Kstate[0]
        rate[21] = rate[20] + 3 * alpha_n(V_rel) * Kstate[1]
        rate[22] = rate[21] + 2 * alpha_n(V_rel) * Kstate[2]
        rate[23] = rate[22] + 1 * alpha_n(V_rel) * Kstate[3]
        rate[24] = rate[23] + 4 * beta_n(V_rel)  * Kstate[4]
        rate[25] = rate[24] + 3 * beta_n(V_rel)  * Kstate[3]
        rate[26] = rate[25] + 2 * beta_n(V_rel)  * Kstate[2]
        rate[27] = rate[26] + 1 * beta_n(V_rel)  * Kstate[1]

        totalrate = rate[27]
        if totalrate <= 1e-9: # Avoid division by zero for very small rates
            break

        # Exponential waiting time
        tupdate = -np.log(np.random.rand()) / totalrate
        tswitch += tupdate
        if tswitch >= (t + dt):
            break

        # Determine which transition occurs
        r = totalrate * np.random.rand()

        if r < rate[0]:    Nastate[0,0] -= 1; Nastate[1,0] += 1
        elif r < rate[1]:  Nastate[1,0] -= 1; Nastate[2,0] += 1
        elif r < rate[2]:  Nastate[2,0] -= 1; Nastate[3,0] += 1
        elif r < rate[3]:  Nastate[3,0] -= 1; Nastate[2,0] += 1
        elif r < rate[4]:  Nastate[2,0] -= 1; Nastate[1,0] += 1
        elif r < rate[5]:  Nastate[1,0] -= 1; Nastate[0,0] += 1
        elif r < rate[6]:  Nastate[0,0] -= 1; Nastate[0,1] += 1
        elif r < rate[7]:  Nastate[1,0] -= 1; Nastate[1,1] += 1
        elif r < rate[8]:  Nastate[2,0] -= 1; Nastate[2,1] += 1
        elif r < rate[9]:  Nastate[3,0] -= 1; Nastate[3,1] += 1
        elif r < rate[10]: Nastate[0,1] -= 1; Nastate[0,0] += 1
        elif r < rate[11]: Nastate[1,1] -= 1; Nastate[1,0] += 1
        elif r < rate[12]: Nastate[2,1] -= 1; Nastate[2,0] += 1
        elif r < rate[13]: Nastate[3,1] -= 1; Nastate[3,0] += 1
        elif r < rate[14]: Nastate[0,1] -= 1; Nastate[1,1] += 1
        elif r < rate[15]: Nastate[1,1] -= 1; Nastate[2,1] += 1
        elif r < rate[16]: Nastate[2,1] -= 1; Nastate[3,1] += 1
        elif r < rate[17]: Nastate[3,1] -= 1; Nastate[2,1] += 1
        elif r < rate[18]: Nastate[2,1] -= 1; Nastate[1,1] += 1
        elif r < rate[19]: Nastate[1,1] -= 1; Nastate[0,1] += 1
        elif r < rate[20]: Kstate[0] -= 1; Kstate[1] += 1
        elif r < rate[21]: Kstate[1] -= 1; Kstate[2] += 1
        elif r < rate[22]: Kstate[2] -= 1; Kstate[3] += 1
        elif r < rate[23]: Kstate[3] -= 1; Kstate[4] += 1
        elif r < rate[24]: Kstate[4] -= 1; Kstate[3] += 1
        elif r < rate[25]: Kstate[3] -= 1; Kstate[2] += 1
        elif r < rate[26]: Kstate[2] -= 1; Kstate[1] += 1
        else:              Kstate[1] -= 1; Kstate[0] += 1

    return Nastate, Kstate

# -----------------------------------------------------------------------------
# Main High-Fidelity HH Simulation Function
# -----------------------------------------------------------------------------
def markov_hh(
    t, Ifunc, NNa, NK, NoiseModel="MarkovChain", Area_for_C=1.0,
    gNa=120.0, gK=36.0, gL=0.3
):
    """
    Returns Y: array with columns [t, V, NaFraction, KFraction]
    """
    dt = t[1] - t[0]
    nt = len(t)
    V_rest = -65.0

    C = 1.0 * Area_for_C
    E_Na = 115.0 + V_rest
    E_K = -12.0 + V_rest
    E_L = 10.6 + V_rest

    m0 = x_inf(alpha_m, beta_m, 0.0)
    h0 = x_inf(alpha_h, beta_h, 0.0)
    n0 = x_inf(alpha_n, beta_n, 0.0)
    V0 = V_rest

    MCNa = np.zeros((4,2))
    MCNa[0,0] = round(NNa * (1-m0)**3 * (1-h0))
    MCNa[1,0] = round(NNa * 3*m0*(1-m0)**2 * (1-h0))
    MCNa[2,0] = round(NNa * 3*m0**2*(1-m0) * (1-h0))
    MCNa[3,0] = round(NNa * m0**3 * (1-h0))
    MCNa[0,1] = round(NNa * (1-m0)**3 * h0)
    MCNa[1,1] = round(NNa * 3*m0*(1-m0)**2 * h0)
    MCNa[2,1] = round(NNa * 3*m0**2*(1-m0) * h0)
    MCNa[3,1] = NNa - np.sum(MCNa)

    MCK = np.zeros(5)
    MCK[0] = round(NK * (1-n0)**4)
    MCK[1] = round(NK * 4*n0*(1-n0)**3)
    MCK[2] = round(NK * 6*n0**2*(1-n0)**2)
    MCK[3] = round(NK * 4*n0**3*(1-n0))
    MCK[4] = NK - np.sum(MCK[:4])

    Y = np.zeros((nt, 4))
    Y[0] = [t[0], V0, MCNa[3,1]/NNa if NNa > 0 else 0, MCK[4]/NK if NK > 0 else 0]

    for i in range(1, nt):
        I_ext = Ifunc(t[i-1])
        
        MCNa, MCK = markov_chain_fraction(V0, MCNa, MCK, t[i-1], dt, V_rest)
        NaFraction = MCNa[3,1] / NNa if NNa > 0 else 0
        KFraction  = MCK[4] / NK if NK > 0 else 0

        g_tot = gNa * NaFraction + gK * KFraction + gL
        I_ion = (gNa * NaFraction * E_Na + gK * KFraction * E_K + gL * E_L)
        V_inf = (I_ion + I_ext) / g_tot
        
        # Avoid division by zero if g_tot is zero
        if g_tot > 1e-9:
            tauV  = C / g_tot
            V_new = V_inf + (V0 - V_inf) * _Exp(-dt / tauV)
        else:
            V_new = V0
        
        Y[i] = [t[i], V_new, NaFraction, KFraction]
        V0 = V_new
        
    return Y

# -----------------------------------------------------------------------------
# Compatibility Wrapper Class
# -----------------------------------------------------------------------------
class MarkovHHSimulator:
    """
    A thin wrapper around the procedural `markov_hh` function to maintain
    compatibility with the class-based project architecture.
    """
    def __init__(self,
                 g_na: float = 120.0, g_k: float = 36.0, g_l: float = 0.3,
                 n_na_channels: int = 6000, n_k_channels: int = 1800):
        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l
        self.n_na = n_na_channels
        self.n_k = n_k_channels

    def simulate(self,
                 t_array: np.ndarray,
                 stimulus_current_func: Callable[[float], float]
                 ) -> np.ndarray:
        return markov_hh(
            t=t_array,
            Ifunc=stimulus_current_func,
            NNa=self.n_na,
            NK=self.n_k,
            gNa=self.g_na,
            gK=self.g_k,
            gL=self.g_l
        )
