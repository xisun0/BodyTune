"""Exact-diameter Performer capacity control on immutable saved circle splits."""

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

from .models import DiameterPerformer, ModelConfig
from .train import evaluate, objective


def load_splits(source):
    splits, hashes = {}, {}
    for name in ("train", "val", "test"):
        path = Path(source) / f"{name}_demonstrations.npz"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            if set(data["family"].tolist()) != {"circle"}:
                raise ValueError("expected circle-only saved data")
            diameter = torch.tensor(data["diameter"], dtype=torch.float32)
            target = torch.tensor(data["q_target"], dtype=torch.float32)
            if target.shape[1:] != (96, 2) or float(data["dt"]) != 0.05:
                raise ValueError("expected 96-sample, dt=.05 protocol")
            splits[name] = (diameter, target)
    return splits, hashes


@torch.inference_mode()
def predict(model, diameter, target):
    model.eval()
    return torch.cat(
        [model(diameter[i : i + 64], target[i : i + 64, 0]) for i in range(0, len(diameter), 64)]
    )


def metrics(q, target, config):
    result = evaluate(q, target, torch.zeros(len(q), 1, dtype=torch.long), config)
    for key in ("distinct_scores", "symbols_used", "mean_length", "length_histogram"):
        result.pop(key)
    return result


def train(
    source,
    output,
    hidden=64,
    epochs=300,
    min_epochs=80,
    patience=40,
    seed=123,
    batch_size=64,
    threads=4,
):
    if hidden not in (64, 128, 256) or min(epochs, batch_size, threads) < 1:
        raise ValueError("invalid training configuration")
    if not 0 <= min_epochs <= epochs or patience < 0:
        raise ValueError("invalid stopping configuration")
    splits, hashes = load_splits(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    config = ModelConfig(tokens=1, hidden=hidden)
    model = DiameterPerformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    diameter, target = splits["train"]
    loader = DataLoader(
        TensorDataset(diameter, target),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    best, best_epoch, history = float("inf"), 0, []
    started = time.monotonic()
    stopping_reason = "epoch_budget"
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for d, b in loader:
            optimizer.zero_grad()
            q = model(d, b[:, 0])
            loss = objective(q, b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(d)
        vd, vb = splits["val"]
        val = metrics(predict(model, vd, vb), vb, config)
        score = val["mean_nrmse"] + 1 - val["feasible_rate"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(diameter),
                "val_nrmse": val["mean_nrmse"],
                "val_feasible": val["feasible_rate"],
                "selection_score": score,
            }
        )
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        checkpoint = {
            "config": asdict(config),
            "model": model.state_dict(),
            "model_type": "exact_diameter_performer",
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "loader_rng": loader.generator.get_state(),
            "history": history,
            "best_score": best,
            "best_epoch": best_epoch,
            "data_sha256": hashes,
        }
        torch.save(checkpoint, output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(json.dumps({"hidden": hidden, **history[-1]}), flush=True)
        if patience and epoch >= min_epochs and epoch - best_epoch >= patience:
            stopping_reason = "validation_patience"
            print(f"hidden={hidden}: stopped at {epoch}, best={best_epoch}", flush=True)
            break
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    td, tb = splits["test"]
    q = predict(model, td, tb)
    # Same within-batch cyclic message permutation as the earlier communication control.
    shuffled = torch.cat([td[i : i + 64].roll(1) for i in range(0, len(td), 64)])
    mean_d = torch.full_like(td, float(diameter.mean()))
    summary = {
        "test": metrics(q, tb, config),
        "shuffled_diameter": metrics(predict(model, shuffled, tb), tb, config),
        "mean_diameter": metrics(predict(model, mean_d, tb), tb, config),
    }
    np.savez_compressed(
        output / "test_rollouts.npz", diameter=td.numpy(), q_target=tb.numpy(), q_pred=q.numpy()
    )
    with (output / "history.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    report = {
        "config": asdict(config),
        "input": "exact scalar diameter in link units; no Notator",
        "training": {
            "seed": seed,
            "epoch_budget": epochs,
            "completed_epochs": epoch,
            "best_epoch": best_epoch,
            "patience": patience,
            "min_epochs": min_epochs,
            "stopping_reason": stopping_reason,
            "batch_size": batch_size,
            "threads": threads,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "gradient_clip": 1.0,
            "seconds": time.monotonic() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
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
            "python": platform.python_version(),
            "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        },
        "limits": "One seed, fixed learning rate, same epoch/stopping protocol, not equal FLOPs. "
        "Hidden width also scales conditioning and action layers. "
        "Accurate scalar diameter is a diagnostic oracle, not a discovered motion language.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"hidden": hidden, "summary": summary}), flush=True)
    return report


def reevaluate(directory, source):
    directory = Path(directory)
    checkpoint = torch.load(directory / "best.pt", weights_only=True)
    splits, hashes = load_splits(source)
    if hashes != checkpoint["data_sha256"]:
        raise ValueError("source dataset hashes changed")
    torch.set_num_threads(4)
    config = ModelConfig(**checkpoint["config"])
    model = DiameterPerformer(config)
    model.load_state_dict(checkpoint["model"])
    d, b = splits["test"]
    return metrics(predict(model, d, b), b, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/circles1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, choices=[64, 128, 256], default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--evaluate", action="store_true")
    args = vars(parser.parse_args())
    evaluate_only = args.pop("evaluate")
    if evaluate_only:
        print(json.dumps(reevaluate(args["output"], args["source"]), indent=2))
    else:
        train(**args)


if __name__ == "__main__":
    main()
