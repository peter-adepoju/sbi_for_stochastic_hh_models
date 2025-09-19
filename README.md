# Efficient Inference of Stochastic Models of Single Neurons with Multifidelity Simulation-Based Inference

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-%234285F4.svg?style=flat)](https://github.com/google/jax)

This repository contains the official source code and results for the Master's thesis titled "Efficient Inference of Stochastic Models of Single Neurons with Multifidelity Simulation-Based Inference" by Peter Oluwafemi Adepoju, submitted in partial fulfillment of a structured master's degree at the African Institute for Mathematical Sciences (AIMS), South Africa.

This work implements and validates a Multifidelity Neural Posterior Estimation (MF-NPE) pipeline for efficiently inferring the biophysical parameters of stochastic Hodgkin-Huxley neuron models.

---

## Abstract

The voltage across the membranes of individual neurons is inherently stochastic, primarily due to the random transitions between conformational states of membrane ion channels. Accurately modeling these phenomena requires computationally intensive simulations, making parameter inference impractical for many neuroscience applications. This project addresses this challenge by developing a multifidelity simulation-based inference (SBI) approach. We combine inexpensive, low-fidelity simulations with a limited budget of costly, high-fidelity simulations to train a neural density estimator. Our results show that this approach significantly outperforms traditional methods, achieving comparable accuracy while requiring orders of magnitude fewer high-fidelity simulations. The study presents a validated, open-source framework that enhances parameter inference in computational neuroscience.

---

## Key Results

Our primary finding is that the multifidelity approach (MF-NPE) can produce posterior distributions that are as accurate (or more accurate) than those from a standard Neural Posterior Estimator (NPE), while using significantly fewer computationally expensive, high-fidelity simulations.

| ![NLTP Comparison](results/nltp_comparison.png) | ![Posterior Comparison](results/posterior_comparison.png) |
|:---:|:---:|
| **Figure 1: NLTP Performance.** Negative Log-Probability of the True Parameters (NLTP, lower is better) vs. number of high-fidelity simulations. MF-NPE variants consistently outperform standard NPE. | **Figure 2: Posterior Accuracy.** Pairwise posteriors for MF-NPE (right) are sharper and more accurate than NPE (left), despite using 10x fewer high-fidelity simulations. |

---

## Repository Structure

The project is organized into a core Python package (`mf_npe`) and several top-level scripts for running experiments.

```
.
├── configs/                  # YAML configuration files for experiments
│   └── hh_experiment.yml
├── mf_npe/                     # The core source code package
│   ├── config/               # Task and plot configuration setup
│   ├── diagnostics/          # Diagnostic plots (PPC, SBC, etc.)
│   ├── flows/                # Building and training normalizing flows
│   ├── plot/                 # High-level plotting functions
│   ├── simulator/            # The LF (JAX) and HF (NumPy) simulators
│   ├── utils/                # Helper functions for stats, I/O, etc.
│   ├── evaluation.py         # The Evaluation class for computing metrics
│   ├── experiment.py         # Logic for a single experimental run
│   └── training.py           # Functions for training inference models
├── sbi/                      # A local copy of the sbi (simulation-based inference) library
├── results/                  # Key figures and results from the thesis
├── run_experiment.py         # Main script to launch a new experiment
├── run_evaluation.py         # Main script to evaluate results and plot figures
└── requirements.txt          # Project dependencies
```

---

## Setup and Installation

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/peter-adepoju/AIMS-Final-project.git
cd AIMS-Final-project
```

### 3. Install Dependencies
It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

---

## How to Run the Experiments

This project is driven by a configuration file (`configs/hh_experiment.yml`). You can modify this file to change hyperparameters, simulation budgets, and models to run.

### Step 1: Run the Training Pipeline
Execute the main experiment script, pointing it to the configuration file. This will generate simulation data (or load it if it exists), train all specified models for each random seed, and save the results.

```bash
python run_experiment.py configs/hh_experiment.yml
```
You can override the seeds specified in the config file from the command line:
```bash
python run_experiment.py configs/hh_experiment.yml --seeds 42 100 2025
```
Outputs for each seed (data, plots, and trained models) will be saved in the directory specified by `output_dir` in the config file (e.g., `./outputs/hh_experiment_final/seed_42/`).

### Step 2: Run the Evaluation Pipeline
Once the training runs are complete, run the evaluation script. This script automatically finds all the results from the experiment, computes the final performance metrics across all seeds, aggregates them, and generates the final summary plots.

```bash
python run_evaluation.py ./outputs/hh_experiment_final/
```
The final aggregated results (as `.csv` files) and summary plots (as `.svg` and `.html` files) will be saved in the `analysis/` sub-directory within your main experiment folder (e.g., `./outputs/hh_experiment_final/analysis/`).

---

## Citing this Work
If you use this code or the ideas presented in this project for your research, please cite the original thesis:

```bibtex
@mastersthesis{Adepoju2025,
  author       = {Adepoju, Peter Oluwafemi},
  title        = {Efficient Inference of Stochastic Models of Single Neurons with Multifidelity Simulation-Based Inference},
  school       = {African Institute for Mathematical Sciences (AIMS) South Africa},
  year         = {2025},
  month        = {July}
}
```

## Acknowledgements
This work was supervised by Prof. Pedro Gonçalves (VIB-Neuroelectronics Research Flanders, KU Leuven, Belgium).
```
