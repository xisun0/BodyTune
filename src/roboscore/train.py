"""CPU-reproducible learning benchmark with validation-selected checkpoints."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import generate
from .kinematics import Arm
from .models import ModelConfig, MotionModel, features, fk


def dataset(count, seed, config, task="mixed", save_path=None):
    pairs, sampling = generate(count, config.steps, seed, config.dt, task=task)
    if save_path is not None:
        payload = {
            "q_a": np.stack([p.q_a for p in pairs]),
            "q_target": np.stack([p.q_b for p in pairs]),
            "relative": np.stack([p.relative for p in pairs]),
            "family": np.array([p.family for p in pairs]),
            "dt": np.array(config.dt),
        }
        if task == "circles":
            payload["diameter"] = np.array([p.diameter for p in pairs])
        np.savez_compressed(save_path, **payload)
    a = torch.tensor(np.stack([p.q_a for p in pairs]), dtype=torch.float32)
    b = torch.tensor(np.stack([p.q_b for p in pairs]), dtype=torch.float32)
    return a, b, [p.family for p in pairs], sampling


def objective(q, target):
    p = fk(q)
    relative = p - p[:, :1]
    desired = fk(target)
    desired = desired - desired[:, :1]
    length = torch.linalg.vector_norm(desired[:, 1:] - desired[:, :-1], dim=-1).sum(-1)
    drawing = ((relative - desired) ** 2).sum(-1).mean(-1) / (length**2).clamp_min(1e-8)
    lower = q.new_tensor(Arm().lower)
    upper = q.new_tensor(Arm().upper)
    constraint = ((lower - q).relu() ** 2 + (q - upper).relu() ** 2).mean()
    singular = (0.05 - q[..., 1].sin().abs()).relu().square().mean()
    dq = q[:, 1:] - q[:, :-1]
    smooth = (dq[:, 1:] - dq[:, :-1]).square().mean()
    return drawing.mean() + constraint + singular + 0.01 * smooth


@torch.no_grad()
def predict(model, a, b, shuffle=False):
    model.eval()
    outputs, scores = [], []
    for start in range(0, len(a), 64):
        batch_a = a[start : start + 64]
        channel, ids = model.notator(features(batch_a, model.config.dt))
        if shuffle:
            # Different demonstration messages, same receiver initial states.
            perm = torch.arange(len(channel)).roll(1)
            ids = ids[perm]
        q = model.decode(ids, b[start : start + 64, 0])
        outputs.append(q)
        scores.append(ids)
    return torch.cat(outputs), torch.cat(scores)


def evaluate(q, b, ids, config):
    arm = Arm()
    predicted = fk(q).detach().numpy()
    target = fk(b).numpy()
    predicted -= predicted[:, :1].copy()
    target -= target[:, :1].copy()
    error = np.linalg.norm(predicted - target, axis=-1)
    length = np.linalg.norm(np.diff(target, axis=1), axis=-1).sum(-1)
    rmse = np.sqrt(np.mean(error**2, axis=1))
    nrmse = rmse / length
    feasible = []
    for path in q.detach().numpy():
        try:
            arm.validate(path, config.dt)
            feasible.append(True)
        except ValueError:
            feasible.append(False)
    feasible = np.array(feasible)
    return {
        "count": len(q),
        "feasible_rate": float(feasible.mean()),
        "success_rate": float(((nrmse <= 0.01) & feasible).mean()),
        "mean_nrmse": float(nrmse.mean()),
        "median_nrmse": float(np.median(nrmse)),
        "mean_rmse": float(rmse.mean()),
        "max_point_error": float(error.max()),
        "distinct_scores": len(np.unique(ids.numpy(), axis=0)),
        "symbols_used": np.unique(
            ids.numpy()[(ids.numpy() >= 0) & (ids.numpy() < config.symbols)]
        ).tolist(),
        "mean_length": float(((ids >= 0) & (ids < config.symbols)).sum(-1).float().mean()),
        "length_histogram": {
            str(int(n)): int(c)
            for n, c in zip(
                *np.unique(
                    ((ids >= 0) & (ids < config.symbols)).sum(-1).numpy(), return_counts=True
                )
            )
        },
    }


def train(
    output,
    epochs=60,
    train_count=1200,
    val_count=240,
    test_count=240,
    seed=123,
    batch_size=64,
    threads=4,
    variable=False,
    tokens=32,
    length_weight=0.01,
    init_checkpoint=None,
    task="mixed",
    patience=0,
    min_epochs=0,
):
    if patience < 0 or min_epochs < 0 or min_epochs > epochs:
        raise ValueError("invalid stopping configuration")
    if min(epochs, train_count, val_count, test_count, batch_size, threads) < 1:
        raise ValueError("counts, epochs, batch size and threads must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    if length_weight < 0:
        raise ValueError("negative length penalty")
    config = ModelConfig(variable=variable, tokens=tokens)
    splits = {
        name: dataset(count, seed + offset, config, task, output / f"{name}_demonstrations.npz")
        for name, count, offset in [
            ("train", train_count, 0),
            ("val", val_count, 1),
            ("test", test_count, 2),
        ]
    }
    a, b, _, _ = splits["train"]
    loader = DataLoader(
        TensorDataset(a, b),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = MotionModel(config)
    initial = None
    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint, weights_only=True)
        old = ModelConfig(**checkpoint["config"])
        if asdict(old) | {"variable": variable} != asdict(config):
            raise ValueError("incompatible initialization configuration")
        weights = checkpoint["model"]
        if variable and not old.variable:
            weights = dict(weights)
            for suffix in ("weight", "bias"):
                key = "notator.head." + suffix
                expanded = model.state_dict()[key].clone()
                expanded[: config.symbols] = weights[key]
                expanded[-1] = 0 if suffix == "weight" else -0.5
                weights[key] = expanded
        model.load_state_dict(weights)
        initial = {
            "path": str(init_checkpoint),
            "epoch": checkpoint["epoch"],
            "sha256": hashlib.sha256(Path(init_checkpoint).read_bytes()).hexdigest(),
        }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, history, best_epoch = float("inf"), [], 0
    started = time.monotonic()
    stopping_reason = "epoch_budget"
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        for a_batch, b_batch in loader:
            optimizer.zero_grad()
            q, _ = model(a_batch, b_batch[:, 0], max(0.4, 1 - epoch / epochs * 0.6))
            loss = objective(q, b_batch)
            if variable:
                loss = loss + length_weight * model.notator.mean_length / config.tokens
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(a_batch)
        va, vb, _, _ = splits["val"]
        q, ids = predict(model, va, vb)
        validation = evaluate(q, vb, ids, config)
        score = validation["mean_nrmse"] + (1 - validation["feasible_rate"])
        if variable:
            score += length_weight * validation["mean_length"] / config.tokens
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / train_count,
                "val_nrmse": validation["mean_nrmse"],
                "selection_score": score,
                "val_length": validation["mean_length"],
            }
        )
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        state = {
            "config": asdict(config),
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "loader_rng": loader.generator.get_state(),
            "history": history,
            "best_score": best,
            "best_epoch": best_epoch,
            "temperature_horizon": epochs,
        }
        torch.save(state, output / "last.pt")
        if improved:
            torch.save(state, output / "best.pt")
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(json.dumps(history[-1]), flush=True)
        if patience and epoch >= min_epochs and epoch - best_epoch >= patience:
            stopping_reason = "validation_patience"
            print(f"Early stop at epoch {epoch}; best validation epoch {best_epoch}", flush=True)
            break
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    ta, tb, families, _ = splits["test"]
    q, ids = predict(model, ta, tb)
    wrong, wrong_ids = predict(model, ta, tb, shuffle=True)
    stationary = tb[:, :1].expand_as(tb)
    summary = {
        "test": evaluate(q, tb, ids, config),
        "shuffled_message": evaluate(wrong, tb, wrong_ids, config),
        "stationary": evaluate(stationary, tb, ids, config),
    }
    np.savez_compressed(
        output / "test_rollouts.npz",
        q_a=ta.numpy(),
        q_target=tb.numpy(),
        q_pred=q.numpy(),
        scores=ids.numpy(),
        family=np.array(families),
    )
    with (output / "history.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    report = {
        "config": asdict(config),
        "training": {
            "epochs": epochs,
            "completed_epochs": epoch,
            "stopping_reason": stopping_reason,
            "patience": patience,
            "min_epochs": min_epochs,
            "task": task,
            "seed": seed,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "threads": threads,
            "seconds": time.monotonic() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "length_weight": length_weight if variable else 0,
            "initialization": initial,
        },
        "splits": {
            name: {"count": len(values[0]), "seed": seed + i, "sampling": values[3]}
            for i, (name, values) in enumerate(splits.items())
        },
        "summary": summary,
        "provenance": {
            "revision": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        },
        "limits": "Independent seeded IID draws within the chosen task, not held-out structural compositions. "
        "No proof of grammar. CPU, unit links, fixed sampling progress. "
        "Straight-through argmax training; hard integer messages at evaluation.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--tokens", type=int, default=32, help="Motion score capacity")
    parser.add_argument("--train-count", type=int, default=1200)
    parser.add_argument("--val-count", type=int, default=240)
    parser.add_argument("--test-count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--variable",
        action="store_true",
        help="8 motion symbols plus EOS, capacity set by --tokens",
    )
    parser.add_argument("--length-weight", type=float, default=0.01)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--task", choices=["mixed", "circles"], default="mixed")
    parser.add_argument(
        "--patience", type=int, default=0, help="Validation early-stop patience; 0 disables"
    )
    parser.add_argument("--min-epochs", type=int, default=0)
    train(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
