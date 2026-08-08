from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .types import ConsensusResult, OperatorSettings


@dataclass
class EnvironmentMoments:
    covariance: Tensor
    cross_moment: Tensor
    count: int


def ridge_head(features: Tensor, targets: Tensor, penalty: float) -> Tensor:
    identity = torch.eye(features.shape[1], device=features.device, dtype=features.dtype)
    return torch.linalg.solve(
        features.transpose(0, 1) @ features + penalty * identity,
        features.transpose(0, 1) @ targets,
    )


def compute_environment_moments(
    features: Tensor,
    targets: Tensor,
    environments: Tensor,
) -> dict[int, EnvironmentMoments]:
    output = {}
    for environment in torch.unique(environments).tolist():
        mask = environments == environment
        local = features[mask]
        local_targets = targets[mask]
        centered = local - local.mean(0)
        target_centered = local_targets - local_targets.mean()
        divisor = max(local.shape[0] - 1, 1)
        covariance = centered.transpose(0, 1) @ centered / divisor
        cross_moment = centered.transpose(0, 1) @ target_centered / divisor
        output[int(environment)] = EnvironmentMoments(
            covariance=covariance,
            cross_moment=cross_moment,
            count=int(local.shape[0]),
        )
    return output


def disagreement_operator(
    moments: dict[int, EnvironmentMoments],
    shared_head: Tensor,
) -> Tensor:
    gradients = []
    weights = []
    for moment in moments.values():
        gradients.append(moment.covariance @ shared_head - moment.cross_moment)
        weights.append(moment.count)
    stacked = torch.stack(gradients)
    centered = stacked - stacked.mean(0)
    operator = centered.transpose(0, 1) @ centered / max(stacked.shape[0] - 1, 1)
    return (operator + operator.transpose(0, 1)) / 2


def invariant_gap(
    features: Tensor,
    targets: Tensor,
    environments: Tensor,
    basis: Tensor,
    penalty: float,
) -> float:
    projected = features @ basis
    shared = ridge_head(projected, targets, penalty)
    worst = 0.0
    for environment in torch.unique(environments).tolist():
        mask = environments == environment
        local_features = projected[mask]
        local_targets = targets[mask]
        local = ridge_head(local_features, local_targets, penalty)
        shared_loss = torch.mean((local_features @ shared - local_targets) ** 2)
        local_loss = torch.mean((local_features @ local - local_targets) ** 2)
        worst = max(worst, float((shared_loss - local_loss).item()))
    return worst


class ConsensusOperator:
    def __init__(
        self,
        settings: OperatorSettings,
        ridge_penalty: float = 1e-4,
        maximum_iterations: int = 100,
    ) -> None:
        self.settings = settings
        self.ridge_penalty = ridge_penalty
        self.maximum_iterations = maximum_iterations

    def fit_basis(
        self,
        features: Tensor,
        targets: Tensor,
        environments: Tensor,
    ) -> tuple[Tensor, Tensor, float]:
        unique_environments = torch.unique(environments)
        if unique_environments.numel() < self.settings.environments:
            raise ValueError("insufficient environments for configured operator")
        moments = compute_environment_moments(features, targets, environments)
        head = ridge_head(features, targets, self.ridge_penalty)
        previous = None
        eigenvalues = torch.empty(0, device=features.device)
        basis = torch.eye(features.shape[1], device=features.device)
        for _ in range(self.maximum_iterations):
            operator = disagreement_operator(moments, head)
            eigenvalues, eigenvectors = torch.linalg.eigh(operator)
            keep = eigenvalues <= self.settings.tolerance
            if not keep.any():
                keep[eigenvalues.argmin()] = True
            basis = eigenvectors[:, keep]
            projected = features @ basis
            projected_head = ridge_head(projected, targets, self.ridge_penalty)
            head = basis @ projected_head
            projector = basis @ basis.transpose(0, 1)
            gap = invariant_gap(
                features,
                targets,
                environments,
                basis,
                self.ridge_penalty,
            )
            if (
                previous is not None
                and torch.allclose(projector, previous, atol=1e-6)
                and gap <= self.settings.tolerance
            ):
                break
            previous = projector
        return basis, eigenvalues, gap

    def node_membership_scores(
        self,
        features: Tensor,
        node_inputs: Tensor,
        environments: Tensor,
        basis: Tensor,
    ) -> Tensor:
        if not node_inputs.requires_grad:
            raise ValueError("node inputs must require gradients")
        projector = basis @ basis.transpose(0, 1)
        projected = features @ projector
        scores = torch.zeros(node_inputs.shape[1], device=node_inputs.device)
        for environment in torch.unique(environments).tolist():
            mask = environments == environment
            scalar = projected[mask].square().sum() / max(int(mask.sum()), 1)
            gradient = torch.autograd.grad(
                scalar,
                node_inputs,
                retain_graph=True,
                create_graph=False,
            )[0]
            scores = scores + gradient[mask].square().sum(dim=(0, 2)).sqrt()
        return scores / torch.unique(environments).numel()

    def permutation_threshold(
        self,
        scores: Tensor,
        generator: torch.Generator,
    ) -> float:
        null_maxima = []
        for _ in range(self.settings.permutations):
            permuted = scores[torch.randperm(scores.numel(), generator=generator)]
            null_maxima.append(permuted.median())
        null = torch.stack(null_maxima)
        return float(torch.quantile(null, self.settings.membership_quantile).item())

    def fit(
        self,
        features: Tensor,
        targets: Tensor,
        environments: Tensor,
        membership_scores: Tensor,
        generator: torch.Generator,
    ) -> ConsensusResult:
        basis, eigenvalues, gap = self.fit_basis(features, targets, environments)
        rank = basis.shape[1]
        if rank >= eigenvalues.numel():
            margin = float("inf")
        else:
            margin = float(eigenvalues[rank].item() - self.settings.tolerance)
        threshold = self.permutation_threshold(membership_scores, generator)
        membership_mask = membership_scores > threshold
        return ConsensusResult(
            basis=basis,
            projector=basis @ basis.transpose(0, 1),
            eigenvalues=eigenvalues,
            rank=rank,
            gap=gap,
            margin=margin,
            membership_scores=membership_scores,
            membership_mask=membership_mask,
        )

    def identifiability_required_environments(
        self,
        representation_dim: int,
        consensus_rank: int,
    ) -> int:
        return representation_dim - consensus_rank + 1

    def certify(
        self,
        result: ConsensusResult,
        representation_dim: int,
        environment_count: int,
    ) -> dict[str, bool | float | int]:
        required = self.identifiability_required_environments(
            representation_dim,
            result.rank,
        )
        return {
            "environment_diversity": environment_count >= required,
            "required_environments": required,
            "observed_environments": environment_count,
            "gap_within_tolerance": result.gap <= self.settings.tolerance,
            "positive_eigenvalue_margin": result.margin >= 0,
            "rank": result.rank,
        }
