# run_experiment.py

"""
The main entry point for launching a full experimental run.
"""

import argparse
from pathlib import Path

import yaml

from mf_npe.config.task_setup import TaskSetup
from mf_npe.experiment import run_single_seed_experiment


def main(args: argparse.Namespace) -> None:
    """
    Loads configuration, sets up, and runs the experiment for all specified seeds.
    """
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    experiment_params = config["experiment"]
    task_params = config["task"]
    model_hyperparams = config["model_hyperparams"]
    
    seeds_to_run = args.seeds if args.seeds else experiment_params.get("seeds", [42])
    print(f"Starting experiment '{experiment_params['name']}' for seeds: {seeds_to_run}")

    for seed in seeds_to_run:
        print(f"\n{'='*20} RUNNING SEED {seed} {'='*20}")

        task_setup = TaskSetup(
            sim_name=task_params["name"],
            main_path=experiment_params["output_dir"],
            lf_sim_budgets=task_params["lf_sim_budgets"],
            hf_sim_budgets=task_params["hf_sim_budgets"],
            config_model=model_hyperparams,
            seed=seed)

        run_single_seed_experiment(task_setup)

    print(f"\n{'='*20} EXPERIMENT COMPLETE {'='*20}")
    print(f"All results saved in: {experiment_params['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a full mf-npe experiment from a configuration file."
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file for the experiment.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional. A list of random seeds to run, overriding the config file.",
    )

    # Example command:
    # python run_experiment.py configs/hh_experiment.yml --seeds 42 43

    parsed_args = parser.parse_args()
    main(parsed_args)
