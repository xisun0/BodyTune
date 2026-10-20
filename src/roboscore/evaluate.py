"""Reload a checkpoint and reproduce hard-score evaluation on its saved test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .models import ModelConfig, MotionModel
from .train import evaluate, predict


def run(directory):
    directory = Path(directory)
    torch.set_num_threads(4)
    checkpoint = torch.load(directory / "best.pt", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = MotionModel(config)
    model.load_state_dict(checkpoint["model"])
    with np.load(directory / "test_rollouts.npz", allow_pickle=False) as data:
        a = torch.from_numpy(data["q_a"].copy())
        b = torch.from_numpy(data["q_target"].copy())
    q, ids = predict(model, a, b)
    return evaluate(q, b, ids, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(run(parser.parse_args().run), indent=2))


if __name__ == "__main__":
    main()
