# mf_npe/training.py

"""
Provides functions for training different types of neural posterior estimators.
"""

from typing import Tuple

import torch
from sbi.inference import SNPE as NPE
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from torch import Tensor, nn
from torch.distributions import Distribution

from mf_npe.config.task_setup import TaskSetup
from mf_npe.flows.build_flows import build_zuko_flow
from mf_npe.flows.train_flows import create_train_val_dataloaders, train_flow


def train_npe(
    thetas: Tensor, xs: Tensor, prior: Distribution, task_setup: TaskSetup
) -> DirectPosterior:
    """
    Trains a standard Neural Posterior Estimator (NPE).

    Args:
        thetas: Training parameters.
        xs: Training summary statistics.
        prior: The prior distribution.
        task_setup: The experiment's TaskSetup object containing hyperparameters.

    Returns:
        The trained posterior object.
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
    Trains a Multifidelity Neural Posterior Estimator (MF-NPE).

    Args:
        thetas_lf, xs_lf: Low-fidelity training data.
        thetas_hf, xs_hf: High-fidelity training data for fine-tuning.
        prior: The prior distribution.
        task_setup: The experiment's TaskSetup object.

    Returns:
        A tuple containing (final_mf_posterior, lf_only_posterior).
    """
    config = task_setup.config_model
    
    # --- Pre-train on Low-Fidelity Data ---
    print("--- Starting LF Pre-training ---")
    lf_flow = build_zuko_flow(
        batch_theta=thetas_lf, batch_x=xs_lf, prior=prior,
        embedding_net=nn.Identity(), z_score_theta=True, z_score_x=True,
        nf_type="NSF_LF", hidden_features=config.get("hidden_features", 50),
        num_transforms=config.get("num_transforms", 5), num_bins=config.get("num_bins", 8),
    )
    lf_optimizer = torch.optim.Adam(lf_flow.parameters(), lr=config.get("learning_rate", 1e-3))
    lf_train_loader, lf_val_loader = create_train_val_dataloaders(
        thetas_lf, xs_lf,
        validation_fraction=config.get("validation_fraction", 0.1),
        batch_size=config.get("batch_size", 50)
    )
    best_lf_flow = train_flow(
        network=lf_flow, optimizer=lf_optimizer,
        train_loader=lf_train_loader, val_loader=lf_val_loader,
        early_stopping_patience=config.get("patience", 20),
        flow_description=f"MF-NPE (LF Pre-training) | LF Sims: {thetas_lf.shape[0]}",
    )
    lf_posterior = DirectPosterior(best_lf_flow, prior)

    # --- Fine-tune on High-Fidelity Data ---
    print("\n--- Starting HF Fine-tuning ---")
    hf_flow = build_zuko_flow(
        batch_theta=thetas_hf, batch_x=xs_hf, prior=prior,
        embedding_net=nn.Identity(), z_score_theta=True, z_score_x=True,
        nf_type="NSF_HF", base_model=best_lf_flow,
    )
    
    # Use a smaller, configurable learning rate for fine-tuning.
    base_lr = config.get("learning_rate", 1e-3)
    lr_factor = config.get("finetuning_lr_factor", 10.0)
    hf_optimizer = torch.optim.Adam(hf_flow.parameters(), lr=base_lr / lr_factor)
    
    hf_train_loader, hf_val_loader = create_train_val_dataloaders(
        thetas_hf, xs_hf,
        validation_fraction=config.get("validation_fraction", 0.1),
        batch_size=config.get("batch_size", 50)
    )
    best_hf_flow = train_flow(
        network=hf_flow, optimizer=hf_optimizer,
        train_loader=hf_train_loader, val_loader=hf_val_loader,
        early_stopping_patience=config.get("patience", 20),
        flow_description=f"MF-NPE (HF Fine-tuning) | HF Sims: {thetas_hf.shape[0]}",
    )
    mf_posterior = DirectPosterior(best_hf_flow, prior)

    return mf_posterior, lf_posterior


def train_sbi_npe(
    thetas: Tensor, xs: Tensor, prior: Distribution, task_setup: TaskSetup
) -> DirectPosterior:
    """
    Trains an NPE using the off-the-shelf sbi library interface for benchmarking.
    
    Args:
        thetas: Training parameters.
        xs: Training summary statistics.
        prior: The prior distribution.
        task_setup: The experiment's TaskSetup object containing hyperparameters.

    Returns:
        The trained posterior object.
    """
    config = task_setup.config_model
    density_estimator_config = {
        "model": "zuko_nsf",
        "hidden_features": config.get("hidden_features", 50),
        "num_transforms": config.get("num_transforms", 5),
        "num_bins": config.get("num_bins", 8)}
    
    inference = NPE(prior=prior, density_estimator=density_estimator_config)
    inference = inference.append_simulations(thetas, xs)
    
    print("Training off-the-shelf SBI NPE...")
    density_estimator = inference.train(
        training_batch_size=config.get("batch_size", 50),
        learning_rate=config.get("learning_rate", 1e-3),
        validation_fraction=config.get("validation_fraction", 0.1),
        stop_after_epochs=config.get("patience", 20),
        max_num_epochs=config.get("max_num_epochs", 2**10),
        show_train_summary=True)
    return inference.build_posterior(density_estimator)