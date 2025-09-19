# run_evaluation.py

"""
A command-line script to run the final evaluation and plotting pipeline.

This script takes an experiment output directory, finds all the results from
individual seeds, evaluates the performance of the trained posteriors, aggregates
the results, and generates the final performance plots.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch

from mf_npe.evaluation import Evaluation
from mf_npe.plot.method_performance import plot_performance_vs_hf_sims
from mf_npe.utils.statistics import calculate_mean_and_ci
from mf_npe.utils.utils import load_pickle


def find_results_files(experiment_dir: Path) -> List[Path]:
    """Finds all per-seed 'results.pkl' files within an experiment directory."""
    print(f"Searching for result files in: {experiment_dir}")
    results_files = list(experiment_dir.glob("**/results.pkl"))
    if not results_files:
        raise FileNotFoundError(f"No 'results.pkl' files found in {experiment_dir}")
    print(f"Found {len(results_files)} result files.")
    return results_files


def evaluate_all_seeds(results_files: List[Path]) -> pd.DataFrame:
    """
    Loads results from each seed, runs evaluation, and returns a combined DataFrame.
    """
    all_seed_dfs = []
    for file_path in results_files:
        print(f"Loading and evaluating results from: {file_path}")
        results = load_pickle(file_path)
        task_setup = results["task_setup"]
        ground_truth = results["ground_truth"]
        
        evaluator = Evaluation(
            true_thetas=ground_truth["thetas"],
            true_xs=ground_truth["xs"],
            evaluation_metric=task_setup.config_data.get("evaluation_metric", "nltp"))
        
        df_seed = evaluator.run_all_evaluations(results["posteriors"])
        df_seed["seed"] = task_setup.seed
        all_seed_dfs.append(df_seed)
        
    return pd.concat(all_seed_dfs, ignore_index=True)


def summarize_results(full_df: pd.DataFrame) -> pd.DataFrame:
    """

    Aggregates results across seeds, calculating mean and confidence intervals.
    """
    if full_df.empty:
        return pd.DataFrame()

    grouping_cols = ["fidelity", "n_lf_simulations", "n_hf_simulations"]
    
    summary_list = []
    for group_keys, group_df in full_df.groupby(grouping_cols):
        metric_values_across_seeds = torch.tensor(group_df["mean"].values)
        mean, lower_ci, upper_ci = calculate_mean_and_ci(metric_values_across_seeds)
        
        summary_row = {
            "fidelity": group_keys[0],
            "n_lf_simulations": group_keys[1],
            "n_hf_simulations": group_keys[2],
            "mean": mean,
            "ci_min": lower_ci,
            "ci_max": upper_ci}
        summary_list.append(summary_row)
        
    return pd.DataFrame(summary_list)


def main(args: argparse.Namespace):
    """Main execution function."""
    experiment_dir = Path(args.experiment_dir)
    
    results_files = find_results_files(experiment_dir)
    full_results_df = evaluate_all_seeds(results_files)
    summary_df = summarize_results(full_results_df)

    # --- Save Results ---
    output_path = experiment_dir / "analysis"
    output_path.mkdir(exist_ok=True)
    full_results_df.to_csv(output_path / "full_evaluation_results.csv", index=False)
    summary_df.to_csv(output_path / "summary_evaluation_results.csv", index=False)
    print(f"\nSaved full and summary results to: {output_path}")

    # --- Plotting ---
    first_result = load_pickle(results_files[0])
    task_setup = first_result["task_setup"]
    
    task_setup._main_path = output_path 
    
    plot_performance_vs_hf_sims(
        df=summary_df,
        lf_simulations=task_setup.lf_sim_budgets,
        evaluation_metric=task_setup.config_data.get("evaluation_metric", "nltp"),
        task_setup=task_setup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate experiment results and generate summary plots."
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="The root directory of the experiment containing the per-seed output folders.",
    )
    
    # Example command:
    # python run_evaluation.py outputs/hh_experiment_final/
    
    parsed_args = parser.parse_args()
    main(parsed_args)