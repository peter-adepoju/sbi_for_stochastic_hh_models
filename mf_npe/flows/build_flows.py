# mf_npe/flows/build_flows.py

"""
Utility functions for building and configuring Zuko-based normalizing flows
for simulation-based inference (SBI), including support for multifidelity models.
"""

import logging
from typing import Union

import torch
import torch.nn as nn
import zuko
from sbi.neural_nets.estimators.zuko_flow import ZukoFlow
from sbi.utils.sbiutils import mcmc_transform
from torch import Tensor
from torch.distributions import Distribution
from zuko.flows import Flow, UnconditionalTransform
from zuko.transforms import AffineTransform


# The standardization logic (Standardize class and z_standardization function) is adapted from the sbi package's utilities to work with the Zuko backend.
# See: https://github.com/sbi-dev/sbi/blob/main/sbi/utils/sbiutils.py


class Standardize(nn.Module):
    """A simple torch module to z-score tensors."""

    def __init__(self, mean: Union[Tensor, float], std: Union[Tensor, float]):
        """
        Initializes the standardization module.
        Args:
            mean: The mean to subtract.
            std: The standard deviation to divide by.
        """
        super().__init__()
        mean, std = map(torch.as_tensor, (mean, std))
        # Register as buffers so they are moved to the correct device (e.g., GPU) with the model, but are not considered model parameters for training.
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, tensor: Tensor) -> Tensor:
        """Applies the z-scoring transformation."""
        return (tensor - self._mean) / self._std


def z_standardization(
    batch_t: Tensor,
    structured_dims: bool = False,
    min_std: float = 1e-14,
) -> tuple[Tensor, Tensor]:
    """
    Computes mean and standard deviation for z-scoring from a batch of data.

    Args:
        batch_t: Batched tensor from which to compute statistics.
        structured_dims: If True, treats dimensions as structured (e.g., time-series)
                         and computes a single mean and std for the entire batch.
                         If False (default), z-scores each feature dimension independently.
        min_std: A minimum value to clamp the standard deviation to, preventing
            division by zero.

    Returns:
        A tuple containing the computed mean and standard deviation.
    """
    if batch_t.ndim == 1:
        batch_t = batch_t.unsqueeze(0)

    if structured_dims:
        t_mean = torch.mean(batch_t)
        sample_std = torch.std(batch_t, dim=1)
        sample_std.clamp_(min=min_std)
        t_std = torch.mean(sample_std)
    else:
        t_mean = torch.mean(batch_t, dim=0)
        t_std = torch.std(batch_t, dim=0)
        t_std.clamp_(min=min_std)

    if batch_t.shape[0] < 2:
        logging.warning(
            "Batch size is < 2. Standardization statistics will not be representative. "
            "Using std=1.0."
        )
        t_std = torch.ones_like(t_std)

    return t_mean, t_std


def standardizing_transform_zuko(
    batch_t: Tensor,
    structured_dims: bool = False,
    min_std: float = 1e-14,
) -> UnconditionalTransform:
    """
    Builds a z-scoring AffineTransform for use in Zuko flows.

    Args:
        batch_t: Batched tensor from which mean and std deviation are computed.
        structured_dims: See `z_standardization`.
        min_std:  See `z_standardization`.

    Returns:
         A Zuko UnconditionalTransform that performs z-scoring.
    """
    t_mean, t_std = z_standardization(batch_t, structured_dims, min_std)

    nan_in_stats = torch.any(torch.isnan(t_mean)) or torch.any(torch.isnan(t_std))
    if nan_in_stats:
        raise ValueError(
            "Training data mean or std for standardizing net must not contain NaNs."
        )

    return UnconditionalTransform(
        AffineTransform,
        loc=-t_mean / t_std,
        scale=1 / t_std,
        buffer=True)


def build_zuko_flow(
    batch_theta: Tensor,
    batch_x: Tensor,
    embedding_net: nn.Module,
    z_score_theta: bool,
    z_score_x: bool,
    prior: Distribution,
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 8,
    nf_type: str = "NSF",
    base_model: Flow = None,
    device: str = "cpu",
) -> ZukoFlow:
    """
    Constructs a Zuko-based normalizing flow for density estimation.

    This function handles standard and multifidelity setups.

    Args:
        batch_theta: A batch of parameters (`theta`) for shape inference.
        batch_x: A batch of data (`x`) for shape inference and standardization.
        embedding_net: The neural network to process `x` and generate context for the flow.
        z_score_theta: Whether to apply z-scoring to the parameters `theta`.
        z_score_x: Whether to apply z-scoring to the data `x`.
        prior: The prior distribution for the parameters. Used to define bounds for the logit transform if applicable.
        hidden_features: The number of hidden units in the transforms.
        num_transforms: The number of transform layers in the flow.
        num_bins: The number of bins for Neural Spline Flows (NSF).
        nf_type: The type of normalizing flow. Supports "NSF" (standard), "NSF_LF" (low-fidelity pre-training), and "NSF_HF" (high-fidelity fine-tuning).
        base_model: A pre-trained flow, required when `nf_type` is "NSF_HF".
        device: The torch device to move the model to.

    Returns:
        A `ZukoFlow` object ready for training.

    Raises:
        NotImplementedError: If an unsupported `nf_type` is provided.
        ValueError: If `base_model` is not provided for "NSF_HF".
    """
    theta_numel = batch_theta.shape[-1]
    embedded_x_numel = embedding_net(batch_x).shape[-1]

    # The core of the flow is a Neural Spline Flow (NSF).
    if nf_type in ("NSF", "NSF_LF"):
        flow_core = zuko.flows.NSF(
            features=theta_numel,
            context=embedded_x_numel,
            bins=num_bins,
            transforms=num_transforms,
            hidden_features=[hidden_features] * num_transforms)

    elif nf_type == "NSF_HF":
        # For high-fidelity fine-tuning, reuse the transforms from the pre-trained model.
        if base_model is None:
            raise ValueError("A `base_model` must be provided for NSF_HF fine-tuning.")

        # We re-use all the core NSF transforms from the pre-trained model.
        # Any pre-processing layers from the original model are not part of the `base_model.net.transform.transforms` list, so we can use them directly.
        pre_trained_transforms = base_model.net.transform.transforms

        flow_core = zuko.flows.Flow(transform=pre_trained_transforms, base=base_model.net.base)

    else:
        raise NotImplementedError(f"Normalizing flow type '{nf_type}' not implemented.")

    # A list to hold all transformations, starting with the core flow's transforms.
    # We will prepend pre-processing transforms to this list.
    all_transforms = list(flow_core.transform.transforms)

    # The order is important: theta is first transformed to be unconstrained (logit), then standardized (z-score).
    if z_score_theta:
        all_transforms.insert(0, standardizing_transform_zuko(batch_theta))

    # Check if the prior has finite support to apply a logit transform.
    if prior.support.is_bounded:
        transform = mcmc_transform(prior)
        # Wrap the sbi transform so it can be used in a Zuko flow.
        all_transforms.insert(0, UnconditionalTransform(lambda: transform, buffer=True))

    if z_score_x:
        # If z-scoring `x`, wrap the embedding_net in a sequence. This computes stats from `batch_x` and creates a standardization layer.
        standardization_layer = Standardize(*z_standardization(batch_x))
        embedding_net = nn.Sequential(standardization_layer, embedding_net)

    # Assemble the final flow with all transformations and the base distribution.
    neural_net = Flow(transform=all_transforms, base=flow_core.base).to(device)

    # Wrap the Zuko flow into an sbi-compatible object. 
    return ZukoFlow(flow=neural_net, embedding_net=embedding_net, theta_shape=batch_theta[0].shape, x_shape=batch_x[0].shape)