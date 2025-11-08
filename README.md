# Efficient Inference of Stochastic Models of Single Neurons with Multifidelity Simulation-Based Inference

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-%234285F4.svg?style=flat)](https://github.com/google/jax)

This repository contains the official source code and results for the Master's thesis titled "Efficient Inference of Stochastic Models of Single Neurons with Multifidelity Simulation-Based Inference" by Peter Oluwafemi Adepoju, submitted in partial fulfillment of a structured master's degree at the African Institute for Mathematical Sciences (AIMS), South Africa, a degree jointly awarded by the University of Cape Town.

This work implements and validates a Multifidelity Neural Posterior Estimation (MF-NPE) pipeline for efficiently inferring the biophysical parameters of stochastic Hodgkin-Huxley neuron models.

---

## Abstract

The voltage across the membranes of individual neurons is inherently stochastic, primarily due to the random transitions between conformational states of membrane ion channels. Accurately modelling these phenomena requires computationally intensive simulations, making parameter inference impractical for many neuroscience applications. This project addresses this challenge by developing a multifidelity simulation-based inference (SBI) approach. We combine inexpensive, low-fidelity simulations with a limited budget of costly, high-fidelity simulations to train a neural density estimator. Our results show that this approach significantly outperforms traditional methods, achieving higher accuracy while requiring orders of magnitude fewer high-fidelity simulations. The study presents a validated, open-source framework that enhances parameter inference in computational neuroscience.

---

## Key Results

Our primary finding is that the multifidelity approach (MF-NPE) can produce posterior distributions that are as accurate (or more accurate) than those from a standard Neural Posterior Estimator (NPE), while using significantly fewer computationally expensive, high-fidelity simulations.

| ![NLTP Comparison](results/nltp_comparison.png) | ![Posterior Comparison](results/posterior_comparison.png) |
|:---:|:---:|
| **Figure 1: NLTP Performance.** Negative Log-Probability of the True Parameters (NLTP, lower is better) vs. Methods. MF-NPE variants consistently outperform standard NPE. | **Figure 2: Posterior Accuracy.** Pairwise posteriors for MF-NPE (right) are sharper and more accurate than for standard NPE (left), despite using 10x fewer high-fidelity simulations. |

---

## Setup and Installation

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/peter-adepoju/sbi_for_stochastic_hh_models.git
cd sbi_for_stochastic_hh_models
```

### 3. Install Dependencies
It is highly recommended to use a virtual environment to manage dependencies.

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the Environment:**
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```

3.  **Install the Required Packages:**
    With the virtual environment active, use `pip` to install all packages listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

---

## Acknowledgements
This work was supervised by Prof. Pedro Gonçalves (VIB-Neuroelectronics Research Flanders, KU Leuven, Belgium).
```
