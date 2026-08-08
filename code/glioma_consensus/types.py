from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

from torch import Tensor


class NodeType(IntEnum):
    IMMUNE = 0
    TUMOUR = 1
    MOLECULAR = 2


class RelationType(IntEnum):
    REGULATORY = 0
    PROTEIN_ASSOCIATION = 1
    CO_METHYLATION = 2
    MOLECULAR_PROGRAMME = 3
    MOLECULAR_IMMUNE = 4
    LIGAND_RECEPTOR = 5
    SPATIAL_ADJACENCY = 6


class Modality(IntEnum):
    GENOMIC = 0
    TRANSCRIPTOMIC = 1
    EPIGENOMIC = 2
    PROTEOMIC = 3
    IMMUNE = 4


class EnvironmentKey(NamedTuple):
    site: int
    region: int
    platform: int
    batch: int


@dataclass(frozen=True)
class ModelDimensions:
    input_dim: int
    hidden_dim: int
    projection_dim: int
    consensus_rank: int
    relation_count: int
    modality_count: int
    layers: int
    heads: int


@dataclass(frozen=True)
class OperatorSettings:
    tolerance: float
    membership_quantile: float
    permutations: int
    environments: int


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    gradient_clip: float
    precision: str


@dataclass
class Prediction:
    probability: Tensor
    uncertainty: Tensor
    abstained: Tensor
    router_weights: Tensor
    projected_representation: Tensor


@dataclass
class ConsensusResult:
    basis: Tensor
    projector: Tensor
    eigenvalues: Tensor
    rank: int
    gap: float
    margin: float
    membership_scores: Tensor
    membership_mask: Tensor
