"""Compare learned modality weighting with concatenation on the frozen TESS split."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import f1_score
import torch

from emotion.data_prep import DATA, HERE
from emotion.experiment import TessSplit, evaluate, load_model, split_fingerprints, train


def compare(args):
    device = torch.device("cpu")
    torch.set_num_threads(4)
    fingerprints = split_fingerprints(args.data)
    dataset = TessSplit(args.data, "test", "fusion")
    results = {}
    for seed in args.seeds:
        baseline = HERE / "checkpoints" if seed == 42 else args.seed_study / f"seed_{seed}"
        weighted = args.work / f"seed_{seed}"
        fit = argparse.Namespace(data=args.data, checkpoints=weighted, batch_size=64,
                                 seed=seed, epochs=30, patience=8, fusion_style="weighted")
        original, checkpoint = load_model("fusion", baseline / "fusion.pt", device)
        if checkpoint.get("data_fingerprints", fingerprints) != fingerprints:
            raise ValueError(f"Seed {seed} baseline data differs from the comparison split")
        truth, scores, _ = evaluate(original, dataset, fit, device, checkpoint["stats"])
        original_prediction = scores.argmax(axis=1)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train("fusion", fit, device)
        model, checkpoint = load_model("fusion", weighted / "fusion.pt", device)
        _, scores, _ = evaluate(model, dataset, fit, device, checkpoint["stats"])
        weighted_prediction = scores.argmax(axis=1)
        original_correct = original_prediction == truth
        weighted_correct = weighted_prediction == truth
        results[str(seed)] = {
            "concat_correct": int(original_correct.sum()),
            "weighted_correct": int(weighted_correct.sum()),
            "concat_macro_f1": float(f1_score(truth, original_prediction, average="macro")),
            "weighted_macro_f1": float(f1_score(truth, weighted_prediction, average="macro")),
            "corrected": int((weighted_correct & ~original_correct).sum()),
            "regressed": int((~weighted_correct & original_correct).sum()),
        }
        print(f"Seed {seed}: concat {original_correct.sum()}/{len(truth)}, weighted {weighted_correct.sum()}/{len(truth)}",
              flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"clips_per_seed": len(dataset), "seeds": results}, indent=2) + "\n",
                               encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    parser.add_argument("--seed-study", type=Path, default=HERE.parent / ".local/seed-study")
    parser.add_argument("--work", type=Path, default=HERE.parent / ".local/weighted-fusion")
    parser.add_argument("--output", type=Path, default=HERE / "Results/weighted_fusion.json")
    compare(parser.parse_args())
