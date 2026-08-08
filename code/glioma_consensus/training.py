from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from .graph import GraphBatch
from .losses import multi_task_loss
from .model import GliomaConsensusNetwork


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    learning_rate: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_learning_rate(
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    maximum: float,
) -> float:
    if epoch < warmup_epochs:
        return maximum * (epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
    return maximum * 0.5 * (1 + np.cos(np.pi * progress))


def atomic_save(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    history: list[EpochRecord],
    destination: Path,
) -> None:
    atomic_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "seed": seed,
            "history": [asdict(item) for item in history],
            "torch_random_state": torch.random.get_rng_state(),
            "numpy_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
        },
        destination,
    )


def restore_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    source: Path,
    device: torch.device,
) -> tuple[int, int, list[EpochRecord]]:
    payload = torch.load(source, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.random.set_rng_state(payload["torch_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    random.setstate(payload["python_random_state"])
    history = [EpochRecord(**item) for item in payload["history"]]
    return int(payload["epoch"]), int(payload["seed"]), history


class Trainer:
    def __init__(
        self,
        model: GliomaConsensusNetwork,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gradient_clip: float,
        router_floor: float,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip = gradient_clip
        self.router_floor = router_floor

    def train_batch(self, batch: GraphBatch) -> Tensor:
        self.model.train()
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model.response_logits(batch)
        survival = self.model.survival_risk(batch)
        _, router_weights = self.model.encode(batch)
        loss = multi_task_loss(
            logits,
            batch.response,
            survival,
            batch.survival_time,
            batch.event,
            router_weights,
            batch.modality_mask,
            self.router_floor,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        return loss.detach()

    def run_epoch(self, batches: Iterable[GraphBatch]) -> float:
        losses = [float(self.train_batch(batch).item()) for batch in batches]
        if not losses:
            raise ValueError("training loader produced no batches")
        return float(np.mean(losses))

    def fit(
        self,
        batches: Iterable[GraphBatch],
        epochs: int,
        maximum_learning_rate: float,
        warmup_epochs: int,
        seed: int,
        output_dir: Path,
    ) -> list[EpochRecord]:
        set_seed(seed)
        history = []
        for epoch in range(epochs):
            learning_rate = cosine_learning_rate(
                epoch,
                epochs,
                warmup_epochs,
                maximum_learning_rate,
            )
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            loss = self.run_epoch(batches)
            history.append(EpochRecord(epoch, loss, learning_rate))
            save_state(
                self.model,
                self.optimizer,
                epoch,
                seed,
                history,
                output_dir / "latest.pt",
            )
            with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
                json.dump([asdict(item) for item in history], handle, indent=2)
        return history
