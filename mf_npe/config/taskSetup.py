# mf_npe/config/task_setup.py

"""
Provides a central configuration object for managing an entire experiment.

The `TaskSetup` class bundles all settings required for a simulation and
inference run, including simulation parameters, model hyperparameters,
and plotting styles.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from mf_npe.config.plot import PLOTTING_STYLE
from mf_npe.utils.task_setup import load_task_config
from mf_npe.utils.utils import set_global_seed


class TaskSetup:
    """
    A central object to hold all configuration for a simulation-inference task.

    Attributes:
        sim_name (str): The name of the simulation task (e.g., 'hh_model').
        config_model (Dict): User-provided hyperparameters for the model/training.
        config_data (Dict): Task-specific constants (e.g., dt, x_dim).
        prior_ranges (Dict): A dictionary of parameter ranges for the prior.
        output_path (Path): The directory for saving experiment results for this seed.
        plotting (Dict): A dictionary containing plotting style settings.
        lf_sim_budgets (List[int]): List of simulation budgets for the LF model.
        hf_sim_budgets (List[int]): List of simulation budgets for the HF model.
    """

    def __init__(
        self,
        sim_name: str,
        config_model: Dict[str, Any],
        main_path: str,
        lf_sim_budgets: List[int],
        hf_sim_budgets: List[int],
        seed: int = 42,
    ):
        """Initializes the complete task configuration for a single seed."""
        self.sim_name: str = sim_name
        self.seed: int = seed
        self.key = set_global_seed(seed)  
        self.timestamp: str = datetime.now().strftime("%Y-%m-%d_%Hh%M")
        
        self.config_model: Dict[str, Any] = config_model
        self.lf_sim_budgets = lf_sim_budgets
        self.hf_sim_budgets = hf_sim_budgets
        
        # The main_path is the top-level directory for the entire experiment.
        self._main_path: Path = Path(main_path)

        self._configure_plotting(PLOTTING_STYLE)
        self._load_task_specifics()

    def _configure_plotting(self, style_dict: Dict[str, Any]) -> None:
        """Loads plotting style constants from the style dictionary."""
        self.plotting = {
            "width": style_dict.get("width_px", 800),
            "height": style_dict.get("height_px", 600),
            "font_size": style_dict.get("label_font_size", 14),
            "title_size": style_dict.get("title_font_size", 16),
            "gridwidth": style_dict.get("grid_linewidth", 1),
            "axis_color": style_dict.get("axis_color", "black"),
            "show_plots": style_dict.get("show_plots_default", False),
        }

    def _load_task_specifics(self) -> None:
        """Loads task-specific data like prior ranges from the config."""
        self.config_data, _, _ = load_task_config(sim_name=self.sim_name)

        # Ensure a default evaluation metric is always present
        self.config_data["evaluation_metric"] = self.config_model.get(
            "evaluation_metric", self.config_data.get("evaluation_metric", "nltp")
        )
        try:
            self.prior_ranges: Dict[str, List[float]] = self.config_data["prior_ranges"]
        except KeyError as e:
            raise ValueError("Task config from utils.task_setup is missing 'prior_ranges' dictionary.") from e

    @property
    def output_path(self) -> Path:
        """
        Returns the seed-specific output path for the experiment.
        e.g., '{main_path}/seed_42/'
        """
        path = self._main_path / f"seed_{self.seed}"
        path.mkdir(parents=True, exist_ok=True)
        return path