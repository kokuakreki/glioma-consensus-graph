from __future__ import annotations

import torch
from torch import Tensor, nn

from .graph import GraphBatch, modality_reachability
from .layers import ModalityExpert, PositiveRouter, segment_mean
from .types import ModelDimensions, Prediction


class GliomaConsensusNetwork(nn.Module):
    def __init__(
        self,
        dimensions: ModelDimensions,
        router_floor: float = 0.01,
        dropout: float = 0.1,
        propagation_hops: int = 3,
    ) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.propagation_hops = propagation_hops
        self.node_type_embedding = nn.Embedding(3, dimensions.input_dim)
        self.modality_embedding = nn.Embedding(
            dimensions.modality_count,
            dimensions.input_dim,
        )
        self.experts = nn.ModuleList(
            ModalityExpert(
                dimensions.input_dim,
                dimensions.hidden_dim,
                dimensions.heads,
                dimensions.relation_count,
                dimensions.layers,
                dropout,
            )
            for _ in range(dimensions.modality_count)
        )
        self.router = PositiveRouter(
            dimensions.hidden_dim,
            dimensions.modality_count,
            router_floor,
        )
        self.projection = nn.Linear(
            dimensions.hidden_dim,
            dimensions.projection_dim,
            bias=False,
        )
        self.response_head = nn.Linear(dimensions.projection_dim, 1)
        self.survival_head = nn.Linear(dimensions.projection_dim, 1)
        self.subtype_head = nn.Linear(dimensions.projection_dim, 3)
        self.register_buffer(
            "consensus_projector",
            torch.eye(dimensions.projection_dim),
        )

    def encode_nodes(self, batch: GraphBatch) -> Tensor:
        features = (
            batch.node_features
            + self.node_type_embedding(batch.node_types)
            + self.modality_embedding(batch.node_modalities)
        )
        pooled = []
        for modality, expert in enumerate(self.experts):
            hidden = expert(
                features,
                batch.edge_index,
                batch.edge_types,
                batch.edge_weights,
                batch.graph_index,
            )
            mask = batch.node_modalities == modality
            masked = hidden * mask[:, None]
            counts = torch.zeros(
                batch.graph_count,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            counts.index_add_(
                0,
                batch.graph_index,
                mask.to(hidden.dtype),
            )
            summary = segment_mean(masked, batch.graph_index, batch.graph_count)
            summary = summary * (counts > 0)[:, None]
            pooled.append(summary)
        return torch.stack(pooled, dim=1)

    def encode(self, batch: GraphBatch) -> tuple[Tensor, Tensor]:
        experts = self.encode_nodes(batch)
        weights = self.router(experts, batch.modality_mask)
        mixture = (experts * weights[:, :, None]).sum(dim=1)
        return self.projection(mixture), weights

    def set_consensus_basis(self, basis: Tensor) -> None:
        if basis.ndim != 2 or basis.shape[0] != self.dimensions.projection_dim:
            raise ValueError("invalid consensus basis")
        projector = basis @ basis.transpose(0, 1)
        self.consensus_projector.copy_(projector)

    def project_consensus(self, representation: Tensor) -> Tensor:
        return representation @ self.consensus_projector

    def modality_dropout_predictions(
        self,
        batch: GraphBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        full_representation, full_weights = self.encode(batch)
        predictions = [
            torch.sigmoid(self.response_head(self.project_consensus(full_representation))).squeeze(
                -1
            )
        ]
        for modality in range(self.dimensions.modality_count):
            altered_mask = batch.modality_mask.clone()
            altered_mask[:, modality] = False
            valid = altered_mask.any(dim=-1)
            if not valid.any():
                continue
            altered = GraphBatch(
                node_features=batch.node_features,
                node_types=batch.node_types,
                node_modalities=batch.node_modalities,
                edge_index=batch.edge_index,
                edge_types=batch.edge_types,
                edge_weights=batch.edge_weights,
                graph_index=batch.graph_index,
                modality_mask=altered_mask,
                environments=batch.environments,
                response=batch.response,
                exposure=batch.exposure,
                survival_time=batch.survival_time,
                event=batch.event,
            )
            representation, _ = self.encode(altered)
            prediction = torch.sigmoid(
                self.response_head(self.project_consensus(representation))
            ).squeeze(-1)
            prediction = torch.where(valid, prediction, predictions[0])
            predictions.append(prediction)
        ensemble = torch.stack(predictions)
        return ensemble.mean(0), ensemble.std(0, unbiased=False), full_weights

    def forward(
        self,
        batch: GraphBatch,
        abstention_threshold: float = 0.12,
    ) -> Prediction:
        probability, uncertainty, weights = self.modality_dropout_predictions(batch)
        representation, _ = self.encode(batch)
        reachable = modality_reachability(
            batch.node_modalities,
            batch.edge_index,
            batch.modality_mask,
            batch.graph_index,
            self.propagation_hops,
        )
        reachable_by_graph = torch.zeros(
            batch.graph_count,
            dtype=torch.bool,
            device=reachable.device,
        )
        reachable_by_graph.scatter_reduce_(
            0,
            batch.graph_index,
            reachable,
            reduce="amin",
            include_self=False,
        )
        abstained = (
            (uncertainty > abstention_threshold)
            | ~batch.modality_mask.any(dim=-1)
            | ~reachable_by_graph
        )
        return Prediction(
            probability=probability,
            uncertainty=uncertainty,
            abstained=abstained,
            router_weights=weights,
            projected_representation=self.project_consensus(representation),
        )

    def response_logits(self, batch: GraphBatch) -> Tensor:
        representation, _ = self.encode(batch)
        return self.response_head(self.project_consensus(representation)).squeeze(-1)

    def survival_risk(self, batch: GraphBatch) -> Tensor:
        representation, _ = self.encode(batch)
        return self.survival_head(self.project_consensus(representation)).squeeze(-1)

    def subtype_logits(self, batch: GraphBatch) -> Tensor:
        representation, _ = self.encode(batch)
        return self.subtype_head(self.project_consensus(representation))
