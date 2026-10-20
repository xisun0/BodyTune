"""Matched GRU128 learned scores and fixed diameter-bin control on saved circles."""

import argparse
import csv
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .capacity import metrics
from .kinematics import Arm
from .models import ModelConfig, Performer
from .train import objective


@dataclass
class AlignedConfig:
    mode: str = "global"
    hidden: int = 128
    encoder_hidden: int = 64
    symbols: int = 8
    steps: int = 96
    dt: float = 0.05
    motion_scale: float = 0.05

    def __post_init__(self):
        if (
            self.mode not in ("global", "local", "binned")
            or self.hidden != 128
            or self.symbols != 8
        ):
            raise ValueError("expected global/local/binned, 128 GRU units and 8 symbols")

    @property
    def tokens(self):
        return self.steps - 1 if self.mode == "local" else 1


class MotionEncoder(nn.Module):
    """Shared local function of endpoint displacement; no joint, position or time input."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        h = config.encoder_hidden
        self.trunk = nn.Sequential(nn.Linear(2, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head = nn.Linear(h, config.symbols)

    def forward(self, motion, temperature=1.0):
        if motion.ndim != 3 or motion.shape[1:] != (self.config.steps - 1, 2):
            raise ValueError("expected one 2D displacement per action step")
        hidden = self.trunk(motion / self.config.motion_scale)
        if self.config.mode == "global":
            hidden = hidden.mean(1, keepdim=True)
        logits = self.head(hidden)
        ids = logits.argmax(-1)
        soft = (logits / temperature).softmax(-1)
        hard = F.one_hot(ids, self.config.symbols).to(soft.dtype)
        return (hard + soft - soft.detach() if self.training else hard), ids


class DiameterBins(nn.Module):
    """Eight fixed equal-width bins on the predeclared [.24, .70] diameter range."""

    def forward(self, diameter, temperature=1.0):
        if diameter.ndim != 1 or not torch.isfinite(diameter).all():
            raise ValueError("expected finite scalar diameters")
        if ((diameter < 0.24) | (diameter > 0.70)).any():
            raise ValueError("diameter outside declared generation range")
        edges = torch.linspace(0.24, 0.70, 9, dtype=diameter.dtype, device=diameter.device)
        ids = torch.bucketize(diameter, edges[1:-1], right=True)[:, None]
        return F.one_hot(ids, 8).float(), ids


class StepPerformer(Performer):
    def __init__(self, config):
        super().__init__(
            ModelConfig(
                hidden=config.hidden,
                symbols=config.symbols,
                steps=config.steps,
                dt=config.dt,
                tokens=1,
            )
        )

    def forward(self, channel, q0):
        if channel.ndim != 3 or channel.shape[1] not in (1, self.config.steps - 1):
            raise ValueError("expected global or step-aligned channel")
        contexts = self.message(channel @ self.embedding)
        return self.rollout(contexts[:, 0] if channel.shape[1] == 1 else contexts, q0)


class AlignedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.notator = MotionEncoder(config)
        self.performer = StepPerformer(config)
        if config.mode == "binned":
            self.notator = DiameterBins()

    def decode(self, scores, q0):
        if scores.dtype != torch.long or scores.ndim != 2 or scores.shape[1] != self.config.tokens:
            raise ValueError("invalid integer motion score")
        if ((scores < 0) | (scores >= self.config.symbols)).any():
            raise ValueError("invalid symbol")
        return self.performer(F.one_hot(scores, self.config.symbols).float(), q0)

    def forward(self, motion, q0, temperature=1.0):
        channel, ids = self.notator(motion, temperature)
        return self.performer(channel, q0), ids


def load_splits(source, mode="global"):
    splits, hashes = {}, {}
    for name in ("train", "val", "test"):
        path = Path(source) / f"{name}_demonstrations.npz"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            if set(data["family"].tolist()) != {"circle"} or float(data["dt"]) != 0.05:
                raise ValueError("expected saved circle protocol")
            # Learned modes read only A endpoint displacements; binned mode reads oracle diameters.
            motion = np.diff(Arm().fk(data["q_a"]), axis=1)
            splits[name] = (
                torch.tensor(data["diameter"] if mode == "binned" else motion, dtype=torch.float32),
                torch.tensor(data["q_target"], dtype=torch.float32),
            )
    return splits, hashes


def weight_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def predict(model, motion, target, control="correct"):
    model.eval()
    outputs, scores = [], []
    for i in range(0, len(motion), 64):
        _, ids = model.notator(motion[i : i + 64])
        if control == "shuffle_demonstrations":
            ids = ids.roll(1, 0)
        elif control == "permute_steps":
            permutation = torch.randperm(ids.shape[1], generator=torch.Generator().manual_seed(777))
            ids = ids[:, permutation]
        elif control != "correct":
            raise ValueError("unknown intervention")
        outputs.append(model.decode(ids, target[i : i + 64, 0]))
        scores.append(ids)
    return torch.cat(outputs), torch.cat(scores)


def evaluate(q, target, scores, config):
    result = metrics(q, target, config)
    result.update(
        distinct_scores=len(np.unique(scores.numpy(), axis=0)),
        symbols_used=np.unique(scores.numpy()).tolist(),
        tokens=config.tokens,
    )
    return result


def train(
    source,
    output,
    mode="global",
    epochs=300,
    min_epochs=80,
    patience=40,
    seed=123,
    batch_size=64,
    threads=4,
):
    if min(epochs, batch_size, threads) < 1 or patience < 0 or not 0 <= min_epochs <= epochs:
        raise ValueError("invalid training/stopping configuration")
    config = AlignedConfig(mode=mode)
    splits, hashes = load_splits(source, mode)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = AlignedModel(config)
    initial_hash = weight_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    motion, target = splits["train"]
    loader = DataLoader(
        TensorDataset(motion, target),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    best, best_epoch, history = float("inf"), 0, []
    reason = "epoch_budget"
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for a, b in loader:
            optimizer.zero_grad()
            q, _ = model(a, b[:, 0], max(0.4, 1 - epoch / epochs * 0.6))
            loss = objective(q, b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(a)
        va, vb = splits["val"]
        q, ids = predict(model, va, vb)
        val = evaluate(q, vb, ids, config)
        score = val["mean_nrmse"] + 1 - val["feasible_rate"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(motion),
                "val_nrmse": val["mean_nrmse"],
                "selection_score": score,
                "val_distinct_scores": val["distinct_scores"],
            }
        )
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        state = {
            "config": asdict(config),
            "model": model.state_dict(),
            "epoch": epoch,
            "model_type": "aligned_motion",
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "loader_rng": loader.generator.get_state(),
            "history": history,
            "best_score": best,
            "best_epoch": best_epoch,
            "temperature_horizon": epochs,
            "data_sha256": hashes,
        }
        torch.save(state, output / "last.pt")
        if improved:
            torch.save(state, output / "best.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(json.dumps({"mode": mode, **history[-1]}), flush=True)
        if patience and epoch >= min_epochs and epoch - best_epoch >= patience:
            reason = "validation_patience"
            print(f"{mode}: stopped at {epoch}, best={best_epoch}", flush=True)
            break
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True)["model"])
    ta, tb = splits["test"]
    q, ids = predict(model, ta, tb)
    summary = {"test": evaluate(q, tb, ids, config)}
    for control in ("shuffle_demonstrations", "permute_steps"):
        wrong, wrong_ids = predict(model, ta, tb, control)
        summary[control] = evaluate(wrong, tb, wrong_ids, config)
    np.savez_compressed(
        output / "test_rollouts.npz",
        motion=ta.numpy(),
        q_target=tb.numpy(),
        q_pred=q.numpy(),
        scores=ids.numpy(),
    )
    with (output / "history.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    report = {
        "config": asdict(config),
        "nominal_bits": config.tokens * 3,
        "diameter_bin_edges": np.linspace(0.24, 0.70, 9).tolist() if mode == "binned" else None,
        "training": {
            "epochs": epochs,
            "completed_epochs": epoch,
            "best_epoch": best_epoch,
            "stopping_reason": reason,
            "min_epochs": min_epochs,
            "patience": patience,
            "seed": seed,
            "batch_size": batch_size,
            "threads": threads,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "gradient_clip": 1.0,
            "seconds": time.monotonic() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "performer_parameters": sum(p.numel() for p in model.performer.parameters()),
            "initial_weights_sha256": initial_hash,
        },
        "data": {
            "source": str(source),
            "sha256": hashes,
            "counts": {name: len(value[0]) for name, value in splits.items()},
        },
        "summary": summary,
        "provenance": {
            "revision": revision,
            "dirty_at_start": dirty,
            "torch": torch.__version__,
            "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        },
        "limits": "Binned mode bypasses the learned Notator using fixed equal-width true-diameter labels; "
        "it estimates achieved performance, not a theoretical upper bound. "
        "Global 3 bits vs local 285 bits. "
        "Shared phase clock and circle-only data. Locality is not isolated from capacity. "
        "No grammar/compositionality claim. Learned modes use straight-through argmax; "
        "binned mode trains only the Performer with fixed integer labels.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"mode": mode, "summary": summary}), flush=True)
    return report


def reevaluate(directory, source):
    directory = Path(directory)
    checkpoint = torch.load(directory / "best.pt", weights_only=True)
    splits, hashes = load_splits(source, checkpoint["config"]["mode"])
    if hashes != checkpoint["data_sha256"]:
        raise ValueError("source hashes changed")
    torch.set_num_threads(4)
    model = AlignedModel(AlignedConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    a, b = splits["test"]
    q, ids = predict(model, a, b)
    return evaluate(q, b, ids, model.config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/circles1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["global", "local", "binned"], default="global")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--evaluate", action="store_true")
    args = vars(parser.parse_args())
    if args.pop("evaluate"):
        print(json.dumps(reevaluate(args["output"], args["source"]), indent=2))
    else:
        train(**args)


if __name__ == "__main__":
    main()
