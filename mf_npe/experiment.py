# mf_npe/experiment.py

"""
Defines and runs a single, self-contained experimental trial.

This module orchestrates the entire process for a single random seed:
1.  Loads or generates the required simulation data (LF, HF, and ground-truth).
2.  Trains all specified inference methods on the data.
3.  Saves the trained posteriors and the configuration for later evaluation.
"""

from pathlib import Path
from typing import Dict, Any, List

import torch
from sbi.utils import simulate_for_sbi
from torch import Tensor

from mf_npe.config.task_setup import TaskSetup
from mf_npe.diagnostics.histogram import plot_summary_statistic_histograms
from mf_npe.simulator.prior import create_prior
from mf_npe.simulator.simulation_func import SimulationWrapper
from mf_npe.training import train_npe, train_mf_npe, train_sbi_npe
from mf_npe.utils.utils import dump_pickle, load_pickle


def generate_or_load_data(task_setup: TaskSetup) -> Dict[str, Any]:
    """
    Generates or loads all necessary data for an experiment.

    This includes low-fidelity, high-fidelity, and ground-truth test data.
    If a data file already exists at the target path, it is loaded; otherwise,
    it is generated and saved.

    Args:
        task_setup: The configured TaskSetup object for the experiment.

    Returns:
        A dictionary containing all data ('lf_data', 'hf_data', 'true_thetas', 'true_xs').
    """
    data_path = task_setup.output_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    
    prior = create_prior(task_setup.prior_ranges)
    
    all_data = {}

    # --- Generate/Load Training Data ---
    for fidelity in ["lf", "hf"]:
        budgets = getattr(task_setup, f"{fidelity}_sim_budgets")
        data_list = []
        for budget in budgets:
            file_path = data_path / f"{fidelity}_data_{budget}.pkl"
            if file_path.exists():
                print(f"Loading existing {fidelity.upper()} data for budget {budget}...")
                data_list.append(load_pickle(file_path))
            else:
                print(f"Generating {fidelity.upper()} data for budget {budget}...")
                sim_config = {"fidelity": fidelity, **task_setup.config_data}
                simulator = SimulationWrapper(config=sim_config, key=task_setup.key)
                thetas, xs = simulate_for_sbi(simulator, prior, num_simulations=budget)
                budget_data = {"thetas": thetas, "xs": xs, "n_samples": budget}
                dump_pickle(file_path, budget_data)
                data_list.append(budget_data)
        all_data[f"{fidelity}_data"] = data_list

    # --- Generate/Load Ground-Truth Test Data ---
    gt_path = data_path / "ground_truth_data.pkl"
    if gt_path.exists():
        print("Loading existing ground-truth data...")
        gt_data = load_pickle(gt_path)
    else:
        print("Generating ground-truth data...")
        num_test = task_setup.config_model.get("num_test_samples", 100)
        hf_sim_config = {"fidelity": "hf", **task_setup.config_data}
        simulator = SimulationWrapper(config=hf_sim_config, key=task_setup.key)
        thetas, xs = simulate_for_sbi(simulator, prior, num_simulations=num_test)
        gt_data = {"true_thetas": thetas, "true_xs": xs}
        dump_pickle(gt_path, gt_data)
    all_data.update(gt_data)

    return all_data


def train_all_methods(
    task_setup: TaskSetup,
    lf_data: List[Dict[str, Tensor]],
    hf_data: List[Dict[str, Tensor]],
) -> Dict[str, Any]:
    """
    Trains all inference methods specified in the task setup.

    Args:
        task_setup: The configured TaskSetup object.
        lf_data: A list of low-fidelity datasets.
        hf_data: A list of high-fidelity datasets.

    Returns:
        A dictionary containing the trained posteriors for each method.
    """
    all_posteriors = {
        "npe": {"posteriors": [], "sim_counts": []},
        "mf_npe": {"posteriors": [], "sim_counts": []},
        "sbi_npe": {"posteriors": [], "sim_counts": []},
    }
    prior = create_prior(task_setup.prior_ranges)

    for method in task_setup.config_model.get("models_to_run", []):
        print(f"\n--- Training method: {method} ---")
        if method == "npe":
            for data_batch in hf_data:
                posterior = train_npe(
                    data_batch["thetas"], data_batch["xs"], prior, task_setup
                )
                all_posteriors["npe"]["posteriors"].append(posterior)
                all_posteriors["npe"]["sim_counts"].append(data_batch["n_samples"])
        
        elif method == "mf_npe":
            for lf_batch in lf_data:
                for hf_batch in hf_data:
                    mf_posterior, _ = train_mf_npe( 
                        lf_batch["thetas"], lf_batch["xs"],
                        hf_batch["thetas"], hf_batch["xs"],
                        prior, task_setup
                    )
                    all_posteriors["mf_npe"]["posteriors"].append(mf_posterior)
                    all_posteriors["mf_npe"]["sim_counts"].append(
                        (lf_batch["n_samples"], hf_batch["n_samples"])
                    )
        
        elif method == "sbi_npe":
            for data_batch in hf_data:
                posterior = train_sbi_npe(
                    data_batch["thetas"], data_batch["xs"], prior, task_setup
                )
                all_posteriors["sbi_npe"]["posteriors"].append(posterior)
                all_posteriors["sbi_npe"]["sim_counts"].append(data_batch["n_samples"])

        else:
            print(f"Warning: Method '{method}' is not recognized in the training script. Skipping.")

    return all_posteriors


def run_single_seed_experiment(task_setup: TaskSetup) -> None:
    """
    Orchestrates a full experiment for a single seed.

    Args:
        task_setup: The fully configured TaskSetup object.
    """
    # Load all necessary data
    data = generate_or_load_data(task_setup)

    # Histograms of the training data
    plots_dir = task_setup.output_path / "plots"
    for fidelity, datasets in [("LF", data["lf_data"]), ("HF", data["hf_data"])]:
        for dataset in datasets:
            plot_summary_statistic_histograms(
                summary_stats=dataset["xs"].numpy(),
                labels=["Spike Count", "Mean Rest", "Std Rest", "Mean Stim"],
                output_path=plots_dir / f"{fidelity}_summaries_{dataset['n_samples']}.html",
                title=f"{fidelity} Training Summaries (n={dataset['n_samples']})",
                plotting_config=task_setup.plotting)

    trained_posteriors = train_all_methods(
        task_setup, data["lf_data"], data["hf_data"])
    
    results = {
        "task_setup": task_setup,
        "posteriors": trained_posteriors,
        "ground_truth": {
            "thetas": data["true_thetas"],
            "xs": data["true_xs"],
        }
    }
    
    results_path = task_setup.output_path / "results.pkl"
    dump_pickle(results_path, results)
    print(f"\nCompleted experiment for seed {task_setup.seed}.")
    print(f"Results saved to {results_path}")