"""Evaluate the three classifiers on a new speaker and new target words."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from emotion.data_prep import DATA, EMOTIONS, HERE
from emotion.experiment import TessSplit, evaluate, load_model, train


def prepare_fold(source, destination, training_speaker, test_speaker):
    destination.mkdir(parents=True, exist_ok=True)
    for split, speaker in (("train", training_speaker), ("validation", training_speaker), ("test", test_speaker)):
        dataset = TessSplit(source, split, "speech")
        indices = [i for i, sample in enumerate(dataset.metadata) if sample["speaker"] == speaker]
        if not indices:
            raise ValueError(f"No {speaker} clips in {split}")
        frames = [dataset[i][0].numpy() for i in indices]
        metadata = [dataset.metadata[i] for i in indices]
        (destination / f"{split}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(destination / f"{split}.npz", frames=np.concatenate(frames),
                            lengths=[len(clip) for clip in frames], labels=dataset.labels[indices].numpy())


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    results = []
    for training_speaker, test_speaker in (("OAF", "YAF"), ("YAF", "OAF")):
        fold = args.work / f"{training_speaker}_to_{test_speaker}"
        data = fold / "data"
        prepare_fold(args.data, data, training_speaker, test_speaker)
        fit = argparse.Namespace(data=data, checkpoints=fold / "checkpoints", batch_size=64,
                                 seed=args.seed, epochs=args.epochs, patience=args.patience)
        scores = {}
        for mode in ("speech", "text", "fusion"):
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            train(mode, fit, device)
            model, checkpoint = load_model(mode, fit.checkpoints / f"{mode}.pt", device)
            truth, probabilities, _ = evaluate(model, TessSplit(data, "test", mode), fit, device, checkpoint["stats"])
            correct = probabilities.argmax(axis=1) == truth
            scores[mode] = {"accuracy": float(correct.mean()),
                            "recall": {emotion: float(correct[truth == i].mean()) for i, emotion in enumerate(EMOTIONS)}}
        results.append({"train_speaker": training_speaker, "test_speaker": test_speaker,
                        "test_clips": len(truth), "scores": scores})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed": args.seed, "epochs": args.epochs,
                                       "patience": args.patience, "folds": results}, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--work", type=Path, default=HERE.parent / ".local/cross-speaker")
    parser.add_argument("--output", type=Path, default=HERE / "Results/cross_speaker.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.threads) < 1:
        parser.error("epochs, patience, and threads must be positive")
    run(args)
