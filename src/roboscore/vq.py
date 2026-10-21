"""Causal TCN/VQ translation of endpoint motion into eight executable symbols."""

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

from .aligned import AlignedConfig, StepPerformer, evaluate, load_splits, weight_hash
from .diagnostics import circle_quality
from .train import objective


@dataclass
class VQConfig:
    hidden: int = 128
    symbols: int = 8
    steps: int = 96
    dt: float = 0.05
    channels: int = 64
    latent: int = 8
    motion_scale: float = 0.05
    commitment: float = 0.25
    auxiliary_weight: float = 0.01
    mode: str = "local"
    auxiliary_clock: bool = False

    def __post_init__(self):
        if self.mode not in ("global", "local"):
            raise ValueError("expected global or local mode")
        if self.mode == "global" and not self.auxiliary_clock:
            raise ValueError("global mode requires the time-conditioned auxiliary decoder")
        if (self.hidden, self.symbols, self.steps, self.latent) != (128, 8, 96, 8):
            raise ValueError("expected GRU128, eight 8D codes and 96 states")
        if self.channels < 1 or self.dt != 0.05 or self.motion_scale <= 0:
            raise ValueError("invalid TCN or motion configuration")

    @property
    def tokens(self):
        return 1 if self.mode == "global" else self.steps - 1


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)

    def forward(self, x):
        return x + F.relu(self.conv(F.pad(x, (self.padding, 0))))


class Translator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input = nn.Conv1d(2, config.channels, 1)
        self.blocks = nn.Sequential(*(CausalBlock(config.channels, d) for d in (1, 2, 4)))
        self.output = nn.Conv1d(config.channels, config.latent, 1)

    def forward(self, motion):
        if motion.ndim != 3 or motion.shape[1:] != (self.config.steps - 1, 2):
            raise ValueError("expected 95 endpoint displacements")
        x = F.relu(self.input((motion / self.config.motion_scale).transpose(1, 2)))
        latent = self.output(self.blocks(x)).transpose(1, 2)
        return latent.mean(1, keepdim=True) if self.config.mode == "global" else latent


class VQModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.translator = Translator(config)
        self.performer = StepPerformer(AlignedConfig(mode="local"))
        # The Performer's existing eight 8D embeddings ARE the shared VQ codebook.
        self.reconstruct = nn.Sequential(
            nn.Linear(10 if config.auxiliary_clock else 8, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def quantize(self, latent):
        codebook = self.performer.embedding
        distance = (latent[..., None, :] - codebook).square().sum(-1)
        ids = distance.argmin(-1)
        selected = F.embedding(ids, codebook)
        quantized = latent + (selected - latent).detach()
        code_loss = F.mse_loss(selected, latent.detach())
        commit_loss = F.mse_loss(latent, selected.detach())
        return quantized, ids, code_loss + self.config.commitment * commit_loss

    def encode(self, motion):
        return self.quantize(self.translator(motion))[1]

    def decode(self, scores, q0):
        if scores.dtype != torch.long or scores.ndim != 2 or scores.shape[1] != self.config.tokens:
            raise ValueError("invalid integer score")
        if ((scores < 0) | (scores >= 8)).any():
            raise ValueError("invalid code ID")
        return self.performer(F.one_hot(scores, 8).float(), q0)

    def reconstruction_loss(self, z, motion):
        if self.config.auxiliary_clock:
            z = z.expand(-1, self.config.steps - 1, -1)
            t = torch.arange(self.config.steps - 1, dtype=z.dtype, device=z.device)
            t = (t / (self.config.steps - 1))[None, :, None].expand(len(z), -1, -1)
            z = torch.cat((z, t, t.square()), -1)
        return F.mse_loss(self.reconstruct(z), motion / self.config.motion_scale)

    def forward(self, motion, q0):
        z, ids, vq_loss = self.quantize(self.translator(motion))
        reconstruction_loss = self.reconstruction_loss(z, motion)
        contexts = self.performer.message(z)
        q = self.performer.rollout(contexts[:, 0] if self.config.tokens == 1 else contexts, q0)
        return q, ids, reconstruction_loss, vq_loss

    @torch.no_grad()
    def initialize_codebook(self, motion):
        """Deterministic train-only k-means++, then Lloyd updates (no semantic labels)."""
        values = torch.cat([self.translator(x).reshape(-1, 8) for x in motion.split(64)])
        generator = torch.Generator().manual_seed(123)
        centers = [values[torch.randint(len(values), (1,), generator=generator)].squeeze(0)]
        for _ in range(7):
            distance = torch.cdist(values, torch.stack(centers)).square().min(-1).values
            index = torch.multinomial(distance.clamp_min(1e-12), 1, generator=generator)
            centers.append(values[index].squeeze(0))
        centers = torch.stack(centers)
        for _ in range(20):
            ids = torch.cdist(values, centers).argmin(-1)
            for code in range(8):
                if (ids == code).any():
                    centers[code] = values[ids == code].mean(0)
        self.performer.embedding.copy_(centers)


@torch.inference_mode()
def predict(model, motion, target, control="correct"):
    model.eval()
    qs, scores = [], []
    for a, b in zip(motion.split(64), target.split(64)):
        ids = model.encode(a)
        if control == "shuffle_demonstrations":
            ids = ids.roll(1, 0)
        elif control == "permute_steps":
            ids = ids[:, torch.randperm(ids.shape[1], generator=torch.Generator().manual_seed(777))]
        elif control != "correct":
            raise ValueError("unknown intervention")
        qs.append(model.decode(ids, b[:, 0]))
        scores.append(ids)
    return torch.cat(qs), torch.cat(scores)


def usage(ids):
    counts = torch.bincount(ids.flatten(), minlength=8)
    p = counts.float() / counts.sum()
    return {
        "counts": counts.tolist(),
        "perplexity": float((-(p * p.clamp_min(1e-12).log()).sum()).exp()),
    }


def train(
    source,
    output,
    epochs=300,
    pretrain_epochs=40,
    min_epochs=80,
    patience=40,
    threads=4,
    mode="local",
    auxiliary_clock=False,
):
    if epochs < 1 or pretrain_epochs < 0 or not 0 <= min_epochs <= epochs or patience < 0:
        raise ValueError("invalid epoch/stopping budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    splits, hashes = load_splits(source, "local")
    torch.set_num_threads(threads)
    torch.manual_seed(123)
    torch.use_deterministic_algorithms(True)
    config = VQConfig(mode=mode, auxiliary_clock=auxiliary_clock)
    model = VQModel(config)
    initial_hash = weight_hash(model)
    model.initialize_codebook(splits["train"][0])
    generator = torch.Generator().manual_seed(123)
    loader = DataLoader(
        TensorDataset(*splits["train"]), batch_size=64, shuffle=True, generator=generator
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    started = time.monotonic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    warmup = []
    # Pretrain translator/codebook/auxiliary decoder only. GRU/message/action layers stay unchanged.
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        total = 0.0
        for a, _ in loader:
            optimizer.zero_grad()
            z, ids, vq_loss = model.quantize(model.translator(a))
            loss = model.reconstruction_loss(z, a) + vq_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(a)
        warmup.append({"epoch": epoch, "loss": total / len(splits["train"][0])})
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"stage": "pretrain", **warmup[-1]}), flush=True)
    # New optimizer and shuffle generator make joint-stage settings explicit and reproducible.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    generator.manual_seed(123)
    history, best, best_epoch = [], float("inf"), 0
    reason = "epoch_budget"
    for epoch in range(1, epochs + 1):
        model.train()
        totals = np.zeros(4)
        for a, b in loader:
            optimizer.zero_grad()
            q, ids, reconstruction, vq_loss = model(a, b[:, 0])
            trajectory = objective(q, b)
            loss = trajectory + config.auxiliary_weight * (reconstruction + vq_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals += np.array(
                [loss.item(), trajectory.item(), reconstruction.item(), vq_loss.item()]
            ) * len(a)
        va, vb = splits["val"]
        q, ids = predict(model, va, vb)
        val = evaluate(q, vb, ids, config)
        score = val["mean_nrmse"] + 1 - val["feasible_rate"]
        row = {
            "epoch": epoch,
            **dict(
                zip(
                    ("loss", "trajectory", "reconstruction", "vq"),
                    (totals / len(splits["train"][0])).tolist(),
                )
            ),
            "val_nrmse": val["mean_nrmse"],
            "val_feasible": val["feasible_rate"],
            "selection_score": score,
            "val_codes": len(val["symbols_used"]),
            "val_perplexity": usage(ids)["perplexity"],
        }
        history.append(row)
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        state = {
            "model_type": "tcn_vq",
            "config": asdict(config),
            "model": model.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_score": best,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "loader_rng": generator.get_state(),
            "history": history,
            "warmup": warmup,
            "data_sha256": hashes,
        }
        torch.save(state, output / "last.pt")
        if improved:
            torch.save(state, output / "best.pt")
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"stage": "joint", **row}), flush=True)
        if patience and epoch >= min_epochs and epoch - best_epoch >= patience:
            reason = "validation_patience"
            break
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True)["model"])
    a, b = splits["test"]
    q, ids = predict(model, a, b)
    summary = {"test": evaluate(q, b, ids, config)}
    for control in ("shuffle_demonstrations", "permute_steps"):
        wrong, wrong_ids = predict(model, a, b, control)
        summary[control] = evaluate(wrong, b, wrong_ids, config)
    with np.load(Path(source) / "test_demonstrations.npz") as data:
        diagnostics = circle_quality(q.numpy(), data["diameter"])
    np.savez_compressed(
        output / "test_rollouts.npz",
        motion=a.numpy(),
        q_target=b.numpy(),
        q_pred=q.numpy(),
        scores=ids.numpy(),
    )
    with (output / "history.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    report = {
        "config": asdict(config),
        "training": {
            "initial_weights_before_kmeans_sha256": initial_hash,
            "seed": 123,
            "batch_size": 64,
            "epochs": epochs,
            "completed_epochs": epoch,
            "best_epoch": best_epoch,
            "pretrain_epochs": pretrain_epochs,
            "min_epochs": min_epochs,
            "patience": patience,
            "stopping_reason": reason,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "gradient_clip": 1.0,
            "threads": threads,
            "seconds": time.monotonic() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "performer_parameters": sum(p.numel() for p in model.performer.parameters()),
        },
        "data": {
            "source": str(source),
            "sha256": hashes,
            "counts": {k: len(v[0]) for k, v in splits.items()},
        },
        "summary": summary,
        "code_usage": usage(ids),
        "circle_diagnostics": diagnostics,
        "provenance": {
            "revision": revision,
            "dirty_at_start": dirty,
            "torch": torch.__version__,
            "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        },
        "limits": f"One seed; circle-only, fixed phase/progress. {config.tokens} tokens/{config.tokens * 3} nominal bits. "
        "TCN receptive field 15 actions; additional 40-epoch reconstruction pretraining. "
        "No diameter/quantizer labels, no continuous latent side channel, no Transformer. "
        "Not a parameter/compute-matched comparison or proof of composition/editing.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def reevaluate(directory, source):
    checkpoint = torch.load(Path(directory) / "best.pt", weights_only=True)
    splits, hashes = load_splits(source, "local")
    if checkpoint["data_sha256"] != hashes:
        raise ValueError("source dataset hashes changed")
    torch.set_num_threads(4)
    model = VQModel(VQConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    a, b = splits["test"]
    q, ids = predict(model, a, b)
    return evaluate(q, b, ids, model.config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/circles1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mode", choices=["global", "local"], default="local")
    parser.add_argument("--auxiliary-clock", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = vars(parser.parse_args())
    if args.pop("evaluate"):
        print(json.dumps(reevaluate(args["output"], args["source"]), indent=2))
    else:
        train(**args)


if __name__ == "__main__":
    main()
