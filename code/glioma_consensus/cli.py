from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .model import GliomaConsensusNetwork
from .types import ModelDimensions


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def dimensions_from_config(config: dict[str, object]) -> ModelDimensions:
    model = config["model"]
    if not isinstance(model, dict):
        raise ValueError("model configuration must be a mapping")
    return ModelDimensions(
        input_dim=int(model["input_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        projection_dim=int(model["projection_dim"]),
        consensus_rank=int(model["consensus_rank"]),
        relation_count=int(model["relation_count"]),
        modality_count=int(model["modality_count"]),
        layers=int(model["layers"]),
        heads=int(model["heads"]),
    )


def train_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/main"))
    args = parser.parse_args()
    config = load_config(args.config)
    model = GliomaConsensusNetwork(dimensions_from_config(config))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "parameter_count": parameter_count,
                "data": str(args.data),
                "config": config,
            },
            handle,
            indent=2,
        )


def evaluate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    with args.results.open("r", encoding="utf-8") as handle:
        json.load(handle)


def prepare_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (args.output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
