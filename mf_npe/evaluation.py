# mf_npe/evaluation.py

"""
Provides a class for evaluating the performance of trained posteriors.

The `Evaluation` class calculates metrics like the Negative Log-Probability of
the True Parameters (NLTP) for a set of posteriors against a ground-truth dataset.
"""

from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from torch import Tensor

from mf_npe.utils.statistics import calculate_mean_and_ci


class Evaluation:
    """
    Handles the evaluation of one or more trained posteriors against test data.
    """
    def __init__(
        self,
        true_thetas: Tensor,
        true_xs: Tensor,
        evaluation_metric: str = "nltp",
    ):
        """
        Initializes the evaluator with the ground-truth dataset.

        Args:
            true_thetas: The ground-truth parameter sets.
            true_xs: The corresponding ground-truth summary statistics (observations).
            evaluation_metric: The name of the metric to compute.
        """
        if evaluation_metric.lower() != "nltp":
            raise NotImplementedError("Only 'nltp' evaluation is currently supported.")
        if torch.isnan(true_xs).any() or torch.isnan(true_thetas).any():
            raise ValueError("Ground-truth data must not contain NaNs.")

        self.true_thetas = true_thetas
        self.true_xs = true_xs
        self.metric = evaluation_metric

    def _evaluate_single_posterior(
        self,
        posterior: NeuralPosterior,
        method_name: str,
        n_sims: Union[int, Tuple[int, int]],
    ) -> pd.DataFrame:
        """
        Computes the NLTP for a single trained posterior.

        Args:
            posterior: The trained sbi posterior object.
            method_name: The name of the method (e.g., 'npe', 'mf_npe').
            n_sims: The number of simulations used for training. Can be an int
                (for HF-only) or a tuple (LF, HF) for multifidelity methods.

        Returns:
            A single-row pandas DataFrame with the evaluation results.
        """
        log_probs = posterior.log_prob(self.true_thetas, self.true_xs)
        nltp_values = -log_probs

        mean, lower_ci, upper_ci = calculate_mean_and_ci(nltp_values)

        # Structure the results into a dictionary
        n_lf, n_hf = (n_sims, 0) if isinstance(n_sims, int) else n_sims
        if "mf" not in method_name and "active" not in method_name:
             n_lf, n_hf = 0, n_sims # HF-only methods

        results = {
            "fidelity": method_name,
            "evaluation_metric": self.metric,
            "n_lf_simulations": n_lf,
            "n_hf_simulations": n_hf,
            "mean": mean,
            "ci_min": lower_ci,
            "ci_max": upper_ci,
            "raw_data": [nltp_values.detach().cpu().numpy()]}
        return pd.DataFrame([results])

    def run_all_evaluations(
        self, all_posteriors: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Evaluates all trained posteriors and compiles the results.

        Args:
            all_posteriors: A nested dictionary structured as:
                {
                    "method_name": {
                        "posteriors": [posterior1, posterior2, ...],
                        "sim_counts": [n_sims1, n_sims2, ...],
                    },
                    ...
                }

        Returns:
            A pandas DataFrame containing the evaluation results for all methods.
        """
        results_list = []
        print("--- Starting Evaluation ---")
        for method_name, data in all_posteriors.items():
            if not data["posteriors"]:
                continue

            print(f"Evaluating method: {method_name}")
            for posterior, n_sims in zip(data["posteriors"], data["sim_counts"]):
                df_row = self._evaluate_single_posterior(
                    posterior, method_name, n_sims
                )
                results_list.append(df_row)

        print("--- Evaluation Complete ---")
        if not results_list:
            return pd.DataFrame()

        return pd.concat(results_list, ignore_index=True)