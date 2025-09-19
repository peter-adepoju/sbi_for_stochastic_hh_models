# mf_npe/flows/train_flows.py

"""
Functions for training and validating normalizing flows for SBI.

This module includes utilities for creating data loaders and a main training
loop with features like early stopping and loss plotting.
"""

import copy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from sbi.neural_nets.estimators.zuko_flow import ZukoFlow
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler


def create_train_val_dataloaders(
    theta: Tensor,
    x: Tensor,
    validation_fraction: float,
    batch_size: int = 200,
) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Creates training and validation dataloaders from simulation data.

    Args:
        theta: A tensor of simulation parameters.
        x: A tensor of corresponding simulation outputs (summary statistics).
        validation_fraction: The fraction of data to use for the validation set.
        batch_size: The batch size for the dataloaders.

    Returns:
        A tuple containing the training DataLoader and the validation DataLoader.
    """
    num_samples = theta.shape[0]
    # This `prior_masks` is a placeholder for future extensions like active learning, where samples might come from different proposals. 
    # For standard training, it is not used in the loss calculation.
    prior_masks = torch.ones(num_samples, 1, dtype=torch.bool)
    dataset = data.TensorDataset(theta, x, prior_masks)

    num_validation_samples = int(validation_fraction * num_samples)
    num_training_samples = num_samples - num_validation_samples

    permuted_indices = torch.randperm(num_samples)
    train_indices = permuted_indices[:num_training_samples]
    val_indices = permuted_indices[num_training_samples:]

    train_loader = data.DataLoader(
                                    dataset,
                                    batch_size=min(batch_size, num_training_samples),
                                    drop_last=True,
                                    sampler=SubsetRandomSampler(train_indices.tolist()),
                                )
    val_loader = data.DataLoader(
                                    dataset,
                                    batch_size=min(batch_size, num_validation_samples),
                                    shuffle=False,  # Sampler handles shuffling
                                    drop_last=True,
                                    sampler=SubsetRandomSampler(val_indices.tolist()),
                                )

    return train_loader, val_loader


def plot_loss(
    training_loss: list, validation_loss: list, flow_description: str
) -> None:
    """
    Plots the training and validation loss curves.

    Args:
        training_loss: A list of training loss values per epoch.
        validation_loss: A list of validation loss values per epoch.
        flow_description: A title for the plot (e.g., 'HF Flow').
    """
    plt.plot(training_loss, label="Training loss")
    plt.plot(validation_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Negative log-likelihood")
    plt.title(flow_description)
    plt.legend()
    plt.show()


def train_flow(
    network: ZukoFlow,
    optimizer: torch.optim.Optimizer,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    max_epochs: int = 2**31 - 1,
    early_stopping_patience: int = 20,
    clip_max_norm: Optional[float] = 5.0,
    show_loss_plot: bool = False,
    flow_description: str = "Flow Training",
    device: str = "cpu",
) -> ZukoFlow:
    """
    Trains a normalizing flow with early stopping.

    Args:
        network: The normalizing flow model to train.
        optimizer: The PyTorch optimizer.
        train_loader: The DataLoader for the training set.
        val_loader: The DataLoader for the validation set.
        max_epochs: The maximum number of epochs to train for.
        early_stopping_patience: Number of epochs to wait for improvement in validation loss before stopping.
        clip_max_norm: The maximum norm for gradient clipping. If None, no clipping.
        show_loss_plot: Whether to display a plot of the loss curves after training.
        flow_description: A string description for the plot title.
        device: The device to train on (e.g., 'cpu' or 'cuda').

    Returns:
        The best performing model on the validation set.
    """
    best_validation_loss = float("inf")
    best_model = copy.deepcopy(network)
    epochs_since_improvement = 0
    training_losses = []
    validation_losses = []

    for epoch in range(max_epochs):
        # --- Training round ---
        network.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            theta_batch, x_batch, _ = (b.to(device) for b in batch)

            losses = -network.log_prob(theta_batch.unsqueeze(0), x_batch)[0]
            loss = torch.mean(losses)
            train_loss_sum += losses.sum().item()

            loss.backward()
            if clip_max_norm is not None:
                clip_grad_norm_(network.parameters(), max_norm=clip_max_norm)
            optimizer.step()

        avg_train_loss = train_loss_sum / len(train_loader.sampler)
        training_losses.append(avg_train_loss)

        # --- Validation round ---
        network.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                theta_batch, x_batch, _ = (b.to(device) for b in batch)
                val_losses = -network.log_prob(theta_batch.unsqueeze(0), x_batch)[0]
                val_loss_sum += val_losses.sum().item()

        avg_val_loss = val_loss_sum / len(val_loader.sampler)
        validation_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1:02d}: Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}",
            end="\r")

        # --- Early stopping ---
        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            best_model = copy.deepcopy(network)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= early_stopping_patience:
            print(f"\nEarly stopping after {epoch + 1} epochs. "
                  f"Best validation loss: {best_validation_loss:.4f}")
            break

    # Clean up gradients from the best model before returning.
    best_model.zero_grad(set_to_none=True)

    if show_loss_plot:
        plot_loss(training_losses, validation_losses, flow_description)

    return best_model