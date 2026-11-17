"""Check how stable the modality comparison is across training seeds, text ablations and noisy audio."""

import argparse
import copy
import json
from pathlib import Path
import random
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from emotion.data_prep import DATA, EMOTIONS, HERE, encode_text, log_mel
from emotion.experiment import TessSplit, evaluate, load_model, train


MODES = ("speech", "text", "fusion")
RELEASED = 42  # The checkpoints in emotion/checkpoints were trained with this seed.


def checkpoint_for(mode, seed, args, device):
    """Train a model for this seed unless it exists. The data split never changes."""
    if seed == RELEASED:
        return HERE / "checkpoints" / f"{mode}.pt"
    directory = args.work / f"seed_{seed}"
    if not (directory / f"{mode}_history.json").exists():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train(mode, argparse.Namespace(**{**vars(args), "checkpoints": directory, "seed": seed}), device)
    return directory / f"{mode}.pt"


def predictions(mode, path, dataset, args, device):
    model, checkpoint = load_model(mode, path, device)
    truth, probabilities, _ = evaluate(model, dataset, args, device, checkpoint["stats"])
    return truth, probabilities.argmax(axis=1)


def with_text(dataset, transcripts):
    changed = copy.copy(dataset)
    changed.tokens = [torch.tensor(encode_text(text)) for text in transcripts]
    return changed


def with_noise(dataset, paths, snr_db):
    rng = np.random.default_rng(0)
    features = [log_mel(paths[item["id"]], snr_db, rng) for item in dataset.metadata]
    changed = copy.copy(dataset)
    changed.frames = torch.from_numpy(np.concatenate(features))
    changed.offsets = np.r_[0, np.cumsum([len(feature) for feature in features])]
    return changed


def spread(values):
    return f"{mean(values):.2%} ± {stdev(values):.2%}" if len(values) > 1 else f"{values[0]:.2%}"


def study(args, device):
    datasets = {mode: TessSplit(args.data, "test", mode) for mode in MODES}
    transcripts = [item["transcript"] for item in datasets["fusion"].metadata]
    shuffled = list(np.random.default_rng(0).permutation(transcripts))
    text_conditions = {"matched transcript": transcripts, "carrier phrase only": ["say the word"] * len(transcripts),
                       "transcript of another clip": shuffled}
    paths = {}
    if args.source:
        paths = {path.stem: path for path in sorted(args.source.rglob("*.wav"))}
        missing = [item["id"] for item in datasets["speech"].metadata if item["id"] not in paths]
        if missing:
            raise ValueError(f"{len(missing)} test clips are missing from {args.source}")
    noisy = {snr: {mode: with_noise(datasets[mode], paths, snr) for mode in ("speech", "fusion")}
             for snr in (args.snr if paths else [])}

    report = {"seeds": {}, "errors_by_emotion": {mode: dict.fromkeys(EMOTIONS, 0) for mode in ("speech", "fusion")},
              "text_ablation": {name: [] for name in text_conditions}, "noise": {}}
    for seed in args.seeds:
        correct = {}
        for mode in MODES:
            path = checkpoint_for(mode, seed, args, device)
            truth, predicted = predictions(mode, path, datasets[mode], args, device)
            correct[mode] = predicted == truth
            if mode in report["errors_by_emotion"]:
                for index, emotion in enumerate(EMOTIONS):
                    report["errors_by_emotion"][mode][emotion] += int((~correct[mode] & (truth == index)).sum())
        report["seeds"][str(seed)] = {
            **{mode: float(correct[mode].mean()) for mode in MODES},
            "fusion_gains": int((correct["fusion"] & ~correct["speech"]).sum()),
            "fusion_losses": int((~correct["fusion"] & correct["speech"]).sum()),
        }
        path = checkpoint_for("fusion", seed, args, device)
        baseline = predictions("fusion", path, datasets["fusion"], args, device)[1]
        for name, texts in text_conditions.items():
            truth, predicted = predictions("fusion", path, with_text(datasets["fusion"], texts), args, device)
            report["text_ablation"][name].append({"accuracy": float((predicted == truth).mean()),
                                                  "changed_predictions": int((predicted != baseline).sum())})
        for snr, variants in noisy.items():
            for mode in ("speech", "fusion"):
                truth, predicted = predictions(mode, checkpoint_for(mode, seed, args, device), variants[mode], args, device)
                report["noise"].setdefault(str(snr), {}).setdefault(mode, []).append(float((predicted == truth).mean()))
        print(f"seed {seed}: " + ", ".join(f"{mode} {report['seeds'][str(seed)][mode]:.4f}" for mode in MODES), flush=True)
    return report, len(transcripts)


def write(report, clips, results):
    seeds = report["seeds"]
    lines = ["# Robustness of the modality comparison", "",
             f"All models use the same word-disjoint split and the same {clips} held-out clips. Only the seed for weight "
             f"initialisation and batch order changes. Seed {RELEASED} is the released checkpoint set.", "",
             "## Training seeds", "",
             "| Seed | Speech | Text | Fusion | Fusion fixes speech errors | Fusion adds errors |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for seed, row in seeds.items():
        lines.append(f"| {seed} | {row['speech']:.2%} | {row['text']:.2%} | {row['fusion']:.2%} | "
                     f"{row['fusion_gains']} | {row['fusion_losses']} |")
    lines.append("| Mean ± SD | " + " | ".join(spread([row[mode] for row in seeds.values()]) for mode in MODES)
                 + f" | {mean(row['fusion_gains'] for row in seeds.values()):.1f} | "
                   f"{mean(row['fusion_losses'] for row in seeds.values()):.1f} |")
    wins = sum(row["fusion"] > row["speech"] for row in seeds.values())
    ties = sum(row["fusion"] == row["speech"] for row in seeds.values())
    lines.extend(["", f"Fusion is more accurate than speech in {wins} of {len(seeds)} seeds, equal in {ties} and "
                      f"less accurate in {len(seeds) - wins - ties}.", "",
                  f"Errors by emotion, summed over the {len(seeds)} seeds ({clips // len(EMOTIONS) * len(seeds)} clips per emotion):", "",
                  "| Model | " + " | ".join(emotion.replace("_", " ").capitalize() for emotion in EMOTIONS) + " |",
                  "| --- |" + " ---: |" * len(EMOTIONS)]
                 + [f"| {mode.capitalize()} | " + " | ".join(str(count) for count in report["errors_by_emotion"][mode].values()) + " |"
                    for mode in report["errors_by_emotion"]]
                 + ["", "## What the fusion model does with the text", "",
                  "The trained fusion models are evaluated again with the text input replaced. The audio is unchanged.", "",
                  "| Text given to the fusion model | Accuracy (mean ± SD) | Predictions that change (mean of 420) |",
                  "| --- | ---: | ---: |"])
    for name, runs in report["text_ablation"].items():
        lines.append(f"| {name} | {spread([run['accuracy'] for run in runs])} | "
                     f"{mean(run['changed_predictions'] for run in runs):.1f} |")
    if report["noise"]:
        lines.extend(["", "## White noise added to the test audio", "",
                      "Noise is added to the 16 kHz signal before trimming and feature extraction. The models were "
                      "trained on clean audio only.", "",
                      "| Signal-to-noise ratio | Speech (mean ± SD) | Fusion (mean ± SD) |", "| --- | ---: | ---: |",
                      f"| clean | {spread([row['speech'] for row in seeds.values()])} | "
                      f"{spread([row['fusion'] for row in seeds.values()])} |"])
        for snr, scores in report["noise"].items():
            lines.append(f"| {float(snr):g} dB | {spread(scores['speech'])} | {spread(scores['fusion'])} |")
        figure, axis = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
        levels = list(report["noise"])
        for mode, color in (("speech", "#2964ac"), ("fusion", "#d97b05")):
            scores = np.array([report["noise"][snr][mode] for snr in levels]) * 100
            axis.plot(range(len(levels)), scores.mean(axis=1), marker="o", color=color, label=mode.title())
            axis.fill_between(range(len(levels)), scores.min(axis=1), scores.max(axis=1), color=color, alpha=0.18)
        axis.axhline(100 / 7, color="#637082", linestyle="--", linewidth=1, label="Uniform guessing")
        axis.set_xticks(range(len(levels)), [f"{float(snr):g} dB" for snr in levels])
        axis.set_xlabel("Signal-to-noise ratio of the test audio")
        axis.set_ylabel("Test accuracy (%)")
        axis.set_title("Accuracy under added white noise (mean and range over seeds)")
        axis.legend()
        (results / "plots").mkdir(parents=True, exist_ok=True)
        figure.savefig(results / "plots" / "robustness.png", dpi=170)
        plt.close(figure)
    (results / "robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (results / "robustness.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--source", type=Path, help="extracted TESS archive; needed for the noise test")
    parser.add_argument("--work", type=Path, default=HERE.parent / ".local/seed-study", help="checkpoints of the extra seeds")
    parser.add_argument("--results", type=Path, default=HERE / "Results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[RELEASED, 1, 2, 3, 4])
    parser.add_argument("--snr", type=float, nargs="+", default=[30, 20, 10, 0])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    outcome, count = study(arguments, torch.device("cpu"))
    write(outcome, count, arguments.results)
