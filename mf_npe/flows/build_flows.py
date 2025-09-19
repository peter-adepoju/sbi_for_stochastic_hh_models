# mf_npe/flows/build_flows.py

"""
Utility functions for building and configuring Zuko-based normalizing flows
for simulation-based inference (SBI), including support for multifidelity models.
"""

import copy
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
    """
    theta_numel = batch_theta.shape[-1]
    embedded_x_numel = embedding_net(batch_x).shape[-1]

    if nf_type in ("NSF", "NSF_LF"):
        flow_core = zuko.flows.NSF(
            features=theta_numel,
            context=embedded_x_numel,
            bins=num_bins,
            transforms=num_transforms,
            hidden_features=[hidden_features] * num_transforms,
        )
    elif nf_type == "NSF_HF":
        if base_model is None:
            raise ValueError("A `base_model` must be provided for NSF_HF fine-tuning.")
        
        # For fine-tuning, create a new instance of the flow with the same architecture as the base model. 
        # We will load the pre-trained weights later.
        lf_flow_transforms = base_model.net.transform.transforms
        flow_core = zuko.flows.Flow(
            transform=[copy.deepcopy(t) for t in lf_flow_transforms],
            base=copy.deepcopy(base_model.net.base),
        )
    else:
        raise NotImplementedError(f"Normalizing flow type '{nf_type}' not implemented.")

    all_transforms = list(flow_core.transform.transforms)

    if z_score_theta:
        all_transforms.insert(0, standardizing_transform_zuko(batch_theta))

    if prior.support.is_bounded:
        transform = mcmc_transform(prior)
        all_transforms.insert(0, UnconditionalTransform(lambda: transform, buffer=True))

    if z_score_x:
        standardization_layer = Standardize(*z_standardization(batch_x))
        embedding_net = nn.Sequential(standardization_layer, embedding_net)

    neural_net = Flow(transform=all_transforms, base=flow_core.base).to(device)
    
    if nf_type == "NSF_HF":
        print("Transferring weights from pre-trained LF model...")
        lf_state_dict = base_model.state_dict()
        neural_net.load_state_dict(lf_state_dict, strict=False)

    return ZukoFlow(
        flow=neural_net,
        embedding_net=embedding_net,
        theta_shape=batch_theta[0].shape,
        x_shape=batch_x[0].shape)
