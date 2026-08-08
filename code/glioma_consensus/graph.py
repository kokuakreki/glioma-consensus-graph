from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from .types import EnvironmentKey, Modality


@dataclass
class GraphCase:
    node_features: Tensor
    node_types: Tensor
    node_modalities: Tensor
    edge_index: Tensor
    edge_types: Tensor
    edge_weights: Tensor
    modality_mask: Tensor
    environment: EnvironmentKey
    response: float
    exposure: float
    survival_time: float
    event: float

    def validate(self) -> None:
        n = self.node_features.shape[0]
        if self.node_features.ndim != 2:
            raise ValueError("node_features must have two dimensions")
        if self.node_types.shape != (n,):
            raise ValueError("node_types length mismatch")
        if self.node_modalities.shape != (n,):
            raise ValueError("node_modalities length mismatch")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        e = self.edge_index.shape[1]
        if self.edge_types.shape != (e,) or self.edge_weights.shape != (e,):
            raise ValueError("edge attribute length mismatch")
        if self.modality_mask.shape != (len(Modality),):
            raise ValueError("modality mask length mismatch")
        if e and (self.edge_index.min() < 0 or self.edge_index.max() >= n):
            raise ValueError("edge index outside node range")


@dataclass
class GraphBatch:
    node_features: Tensor
    node_types: Tensor
    node_modalities: Tensor
    edge_index: Tensor
    edge_types: Tensor
    edge_weights: Tensor
    graph_index: Tensor
    modality_mask: Tensor
    environments: Tensor
    response: Tensor
    exposure: Tensor
    survival_time: Tensor
    event: Tensor

    @property
    def graph_count(self) -> int:
        return int(self.modality_mask.shape[0])

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(
            node_features=self.node_features.to(device),
            node_types=self.node_types.to(device),
            node_modalities=self.node_modalities.to(device),
            edge_index=self.edge_index.to(device),
            edge_types=self.edge_types.to(device),
            edge_weights=self.edge_weights.to(device),
            graph_index=self.graph_index.to(device),
            modality_mask=self.modality_mask.to(device),
            environments=self.environments.to(device),
            response=self.response.to(device),
            exposure=self.exposure.to(device),
            survival_time=self.survival_time.to(device),
            event=self.event.to(device),
        )


def collate_cases(cases: Iterable[GraphCase]) -> GraphBatch:
    items = list(cases)
    if not items:
        raise ValueError("cannot collate an empty collection")
    for case in items:
        case.validate()
    node_features = []
    node_types = []
    node_modalities = []
    edge_indices = []
    edge_types = []
    edge_weights = []
    graph_indices = []
    modality_masks = []
    environments = []
    responses = []
    exposures = []
    survival_times = []
    events = []
    offset = 0
    for graph_id, case in enumerate(items):
        n = case.node_features.shape[0]
        node_features.append(case.node_features)
        node_types.append(case.node_types)
        node_modalities.append(case.node_modalities)
        edge_indices.append(case.edge_index + offset)
        edge_types.append(case.edge_types)
        edge_weights.append(case.edge_weights)
        graph_indices.append(torch.full((n,), graph_id, dtype=torch.long))
        modality_masks.append(case.modality_mask)
        environments.append(torch.tensor(case.environment, dtype=torch.long))
        responses.append(case.response)
        exposures.append(case.exposure)
        survival_times.append(case.survival_time)
        events.append(case.event)
        offset += n
    return GraphBatch(
        node_features=torch.cat(node_features),
        node_types=torch.cat(node_types),
        node_modalities=torch.cat(node_modalities),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_types=torch.cat(edge_types),
        edge_weights=torch.cat(edge_weights),
        graph_index=torch.cat(graph_indices),
        modality_mask=torch.stack(modality_masks),
        environments=torch.stack(environments),
        response=torch.tensor(responses, dtype=torch.float32),
        exposure=torch.tensor(exposures, dtype=torch.float32),
        survival_time=torch.tensor(survival_times, dtype=torch.float32),
        event=torch.tensor(events, dtype=torch.float32),
    )


def prune_edges(
    edge_index: Tensor,
    edge_types: Tensor,
    edge_weights: Tensor,
    node_count: int,
    threshold: float,
) -> tuple[Tensor, Tensor, Tensor]:
    retained = edge_weights >= threshold
    chosen = torch.nonzero(retained, as_tuple=False).flatten().tolist()
    incident = torch.zeros(node_count, dtype=torch.bool, device=edge_index.device)
    if chosen:
        selected = torch.tensor(chosen, device=edge_index.device)
        incident[edge_index[:, selected].reshape(-1)] = True
    for node in torch.nonzero(~incident, as_tuple=False).flatten().tolist():
        candidates = torch.nonzero(
            (edge_index[0] == node) | (edge_index[1] == node),
            as_tuple=False,
        ).flatten()
        if candidates.numel():
            best = candidates[edge_weights[candidates].argmax()]
            chosen.append(int(best))
    if not chosen:
        return edge_index[:, :0], edge_types[:0], edge_weights[:0]
    unique = torch.tensor(sorted(set(chosen)), dtype=torch.long, device=edge_index.device)
    return edge_index[:, unique], edge_types[unique], edge_weights[unique]


def make_bidirectional(
    edge_index: Tensor,
    edge_types: Tensor,
    edge_weights: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    reverse = edge_index.flip(0)
    return (
        torch.cat([edge_index, reverse], dim=1),
        torch.cat([edge_types, edge_types]),
        torch.cat([edge_weights, edge_weights]),
    )


def environment_ids(environments: Tensor) -> Tensor:
    _, inverse = torch.unique(environments, dim=0, return_inverse=True)
    return inverse


def modality_reachability(
    node_modalities: Tensor,
    edge_index: Tensor,
    modality_mask: Tensor,
    graph_index: Tensor,
    hops: int,
) -> Tensor:
    observed = modality_mask[graph_index, node_modalities]
    reachable = observed.clone()
    source, target = edge_index
    for _ in range(hops):
        propagated = torch.zeros_like(reachable)
        propagated.index_add_(0, target, reachable[source].to(torch.long))
        propagated.index_add_(0, source, reachable[target].to(torch.long))
        reachable = reachable | (propagated > 0)
    return reachable
