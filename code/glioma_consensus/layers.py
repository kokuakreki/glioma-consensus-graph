from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def segment_sum(values: Tensor, index: Tensor, count: int) -> Tensor:
    shape = (count,) + values.shape[1:]
    output = values.new_zeros(shape)
    output.index_add_(0, index, values)
    return output


def segment_mean(values: Tensor, index: Tensor, count: int) -> Tensor:
    totals = segment_sum(values, index, count)
    sizes = torch.bincount(index, minlength=count).to(values.dtype)
    shape = (count,) + (1,) * (values.ndim - 1)
    return totals / sizes.clamp_min(1).view(shape)


def segment_softmax(values: Tensor, index: Tensor, count: int) -> Tensor:
    maxima = values.new_full((count, values.shape[1]), -torch.inf)
    maxima.scatter_reduce_(
        0,
        index[:, None].expand_as(values),
        values,
        reduce="amax",
        include_self=True,
    )
    shifted = values - maxima[index]
    exponentials = shifted.exp()
    denominators = segment_sum(exponentials, index, count)
    return exponentials / denominators[index].clamp_min(torch.finfo(values.dtype).tiny)


class RelationTypedAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, relation_count: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden dimension must be divisible by head count")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(relation_count)
        )
        self.value = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(relation_count)
        )
        self.relation_bias = nn.Parameter(torch.zeros(relation_count, heads))
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_types: Tensor,
        edge_weights: Tensor,
    ) -> Tensor:
        source, target = edge_index
        q = self.query(x).view(-1, self.heads, self.head_dim)
        keys = x.new_empty((source.numel(), self.heads, self.head_dim))
        values = torch.empty_like(keys)
        for relation, (key_layer, value_layer) in enumerate(zip(self.key, self.value)):
            mask = edge_types == relation
            if mask.any():
                keys[mask] = key_layer(x[source[mask]]).view(-1, self.heads, self.head_dim)
                values[mask] = value_layer(x[source[mask]]).view(
                    -1,
                    self.heads,
                    self.head_dim,
                )
        scores = (q[target] * keys).sum(-1) / math.sqrt(self.head_dim)
        scores = scores + self.relation_bias[edge_types]
        scores = scores + edge_weights.clamp_min(1e-8).log()[:, None]
        attention = segment_softmax(scores, target, x.shape[0])
        messages = values * self.dropout(attention)[:, :, None]
        aggregated = segment_sum(messages, target, x.shape[0]).flatten(1)
        return self.norm(x + self.output(aggregated))


class SparseGlobalAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, graph_index: Tensor) -> Tensor:
        output = torch.empty_like(x)
        for graph_id in torch.unique(graph_index).tolist():
            mask = graph_index == graph_id
            values = x[mask].unsqueeze(0)
            attended, _ = self.attention(values, values, values, need_weights=False)
            output[mask] = attended.squeeze(0)
        return self.norm(x + output)


class GatedFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int, dropout: float) -> None:
        super().__init__()
        inner = hidden_dim * expansion
        self.gate = nn.Linear(hidden_dim, inner)
        self.value = nn.Linear(hidden_dim, inner)
        self.output = nn.Linear(inner, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        update = torch.nn.functional.silu(self.gate(x)) * self.value(x)
        return self.norm(x + self.output(self.dropout(update)))


class TypedTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        relation_count: int,
        dropout: float,
        global_attention: bool,
    ) -> None:
        super().__init__()
        self.relational = RelationTypedAttention(
            hidden_dim,
            heads,
            relation_count,
            dropout,
        )
        self.global_layer = (
            SparseGlobalAttention(hidden_dim, heads, dropout) if global_attention else None
        )
        self.feed_forward = GatedFeedForward(hidden_dim, 4, dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_types: Tensor,
        edge_weights: Tensor,
        graph_index: Tensor,
    ) -> Tensor:
        x = self.relational(x, edge_index, edge_types, edge_weights)
        if self.global_layer is not None:
            x = self.global_layer(x, graph_index)
        return self.feed_forward(x)


class PositiveRouter(nn.Module):
    def __init__(self, hidden_dim: int, modality_count: int, floor: float) -> None:
        super().__init__()
        if floor * modality_count >= 1:
            raise ValueError("router floor is too large")
        self.modality_count = modality_count
        self.floor = floor
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * modality_count + modality_count, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, modality_count),
        )

    def forward(self, experts: Tensor, modality_mask: Tensor) -> Tensor:
        batch = experts.shape[0]
        logits = self.network(
            torch.cat([experts.reshape(batch, -1), modality_mask.to(experts.dtype)], dim=-1)
        )
        logits = logits.masked_fill(~modality_mask, -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        count = modality_mask.sum(-1, keepdim=True).clamp_min(1)
        available_floor = self.floor * modality_mask.to(weights.dtype)
        scale = 1 - self.floor * count
        return weights * scale + available_floor


class ModalityExpert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        relation_count: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            TypedTransformerBlock(
                hidden_dim,
                heads,
                relation_count,
                dropout,
                global_attention=layer % 2 == 1,
            )
            for layer in range(layers)
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_types: Tensor,
        edge_weights: Tensor,
        graph_index: Tensor,
    ) -> Tensor:
        hidden = self.input_layer(x)
        for block in self.blocks:
            hidden = block(
                hidden,
                edge_index,
                edge_types,
                edge_weights,
                graph_index,
            )
        return hidden
