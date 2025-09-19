# mf_npe/training.py

"""
Provides functions for training different types of neural posterior estimators.
"""
import copy
from typing import Tuple

import numpy as np
import torch
from sbi.inference import SNPE as NPE
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from torch import Tensor, nn
from torch.distributions import Distribution

from mf_npe.config.task_setup import TaskSetup
from mf_npe.flows.build_flows import build_zuko_flow
from mf_npe.flows.train_flows import create_train_val_dataloaders, train_flow
from mf_npe.simulator.simulation_func import SimulationWrapper


def train_npe(
    thetas: Tensor, xs: Tensor, prior: Distribution, task_setup: TaskSetup
) -> DirectPosterior:
    """
    Trains a standard Neural Posterior Estimator (NPE).
    """
    config = task_setup.config_model
    flow = build_zuko_flow(
        batch_theta=thetas, batch_x=xs, prior=prior,
        embedding_net=nn.Identity(),
        z_score_theta=config.get("z_score_theta", True),
        z_score_x=config.get("z_score_x", True),
        nf_type="NSF",
        hidden_features=config.get("hidden_features", 50),
        num_transforms=config.get("num_transforms", 5),
        num_bins=config.get("num_bins", 8),
    )

    optimizer = torch.optim.Adam(flow.parameters(), lr=config.get("learning_rate", 1e-3))
    train_loader, val_loader = create_train_val_dataloaders(
        thetas, xs,
        validation_fraction=config.get("validation_fraction", 0.1),
        batch_size=config.get("batch_size", 50)
    )

    best_flow = train_flow(
        network=flow, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader,
        early_stopping_patience=config.get("patience", 20),
        flow_description=f"NPE | HF Sims: {thetas.shape[0]}",
    )
    return DirectPosterior(best_flow, prior)


def train_mf_npe(
    thetas_lf: Tensor, xs_lf: Tensor,
    thetas_hf: Tensor, xs_hf: Tensor,
    prior: Distribution, task_setup: TaskSetup
) -> Tuple[DirectPosterior, DirectPosterior]:
    """
    Trains a Multifidelity NPE using concatenated summary statistics.
    """
    config = task_setup.config_model
    
    # --- Pre-train a model on Low-Fidelity Data ---
    print("--- Starting LF Pre-training ---")
    lf_posterior = train_npe(thetas_lf, xs_lf, prior, task_setup)
    best_lf_flow = lf_posterior.net

    # --- Prepare the Multifidelity Dataset ---
    print("\n--- Preparing MF Dataset ---")
    print("Simulating LF summaries for each of the HF parameters...")
    
    lf_sim_config = {"fidelity": "lf", **task_setup.config_data}
    lf_simulator_for_hf = SimulationWrapper(config=lf_sim_config, key=task_setup.key)
    
    xs_lf_at_hf_list = [lf_simulator_for_hf(theta) for theta in thetas_hf]
    xs_lf_at_hf = torch.from_numpy(np.stack(xs_lf_at_hf_list)).float()

    # The multifidelity summary statistic is the concatenation of the two.
    xs_mf = torch.cat([xs_lf_at_hf, xs_hf], dim=1)
    
    print(f"Created concatenated MF summary stats with shape: {xs_mf.shape}")

    # --- Fine-tune on the Multifidelity Dataset ---
    print("\n--- Starting MF Fine-tuning ---")
    
    # We create a simple embedding network that takes the concatenated `xs_mf` and projects it down to the original low-fidelity dimension.
    # This allows us to re-use the pre-trained weights of the core flow.
    embedding_net_mf = nn.Linear(xs_mf.shape[1], xs_lf.shape[1])

    mf_flow = build_zuko_flow(
        batch_theta=thetas_hf, batch_x=xs_mf, prior=prior,
        embedding_net=embedding_net_mf,
        z_score_theta=True, z_score_x=True,
        nf_type="NSF_HF", base_model=best_lf_flow,
    )
    
    # Use a smaller, configurable learning rate for fine-tuning.
    base_lr = config.get("learning_rate", 1e-3)
    lr_factor = config.get("finetuning_lr_factor", 10.0)
    mf_optimizer = torch.optim.Adam(mf_flow.parameters(), lr=base_lr / lr_factor)
    
    mf_train_loader, mf_val_loader = create_train_val_dataloaders(
        thetas_hf, xs_mf, # Train with HF thetas and concatenated MF summaries
        validation_fraction=config.get("validation_fraction", 0.1),
        batch_size=config.get("batch_size", 50)
    )
    best_mf_flow = train_flow(
        network=mf_flow, optimizer=mf_optimizer,
        train_loader=mf_train_loader, val_loader=mf_val_loader,
        early_stopping_patience=config.get("patience", 20),
        flow_description=f"MF-NPE Fine-tuning | HF Sims: {thetas_hf.shape[0]}",
    )
    mf_posterior = DirectPosterior(best_mf_flow, prior)

    return mf_posterior, lf_posterior


def train_sbi_npe(
    thetas: Tensor, xs: Tensor, prior: Distribution, task_setup: TaskSetup
) -> DirectPosterior:
    """
    Trains an NPE using the off-the-shelf sbi library interface for benchmarking.
    """
    config = task_setup.config_model
    density_estimator_config = {
        "model": "zuko_nsf",
        "hidden_features": config.get("hidden_features", 50),
        "num_transforms": config.get("num_transforms", 5),
        "num_bins": config.get("num_bins", 8),
    }
    
    inference = NPE(prior=prior, density_estimator=density_estimator_config)
    inference = inference.append_simulations(thetas, xs)
    
    print("Training off-the-shelf SBI NPE...")
    density_estimator = inference.train(
        training_batch_size=config.get("batch_size", 50),
        learning_rate=config.get("learning_rate", 1e-3),
        validation_fraction=config.get("validation_fraction", 0.1),
        stop_after_epochs=config.get("patience", 20),
        max_num_epochs=config.get("max_num_epochs", 2**10),
        show_train_summary=True,
    )
    return inference.build_posterior(density_estimator)
