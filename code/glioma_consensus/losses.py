from __future__ import annotations

import torch
from torch import Tensor


def response_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)


def cox_partial_likelihood(risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
    order = torch.argsort(time, descending=True)
    ordered_risk = risk[order]
    ordered_event = event[order]
    log_cumulative = torch.logcumsumexp(ordered_risk, dim=0)
    terms = (ordered_risk - log_cumulative) * ordered_event
    return -terms.sum() / ordered_event.sum().clamp_min(1)


def interaction_loss(
    risk: Tensor,
    signature: Tensor,
    exposure: Tensor,
    time: Tensor,
    event: Tensor,
) -> Tensor:
    interaction = signature * exposure
    adjusted = risk + interaction
    return cox_partial_likelihood(adjusted, time, event)


def router_floor_penalty(weights: Tensor, mask: Tensor, floor: float) -> Tensor:
    observed = weights[mask]
    if not observed.numel():
        return weights.sum() * 0
    return torch.relu(floor - observed).square().mean()


def multi_task_loss(
    response_logits: Tensor,
    response_targets: Tensor,
    survival_risk: Tensor,
    survival_time: Tensor,
    event: Tensor,
    router_weights: Tensor,
    modality_mask: Tensor,
    router_floor: float,
) -> Tensor:
    classification = response_loss(response_logits, response_targets)
    survival = cox_partial_likelihood(survival_risk, survival_time, event)
    routing = router_floor_penalty(router_weights, modality_mask, router_floor)
    return classification + 0.2 * survival + 0.01 * routing
