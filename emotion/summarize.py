"""Compare held-out predictions and visualize the learned representations."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from emotion.data_prep import EMOTIONS, HERE


MODES = {"speech": "Speech temporal", "text": "Text contextual", "fusion": "Speech + text fusion"}
NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant surprise", "Sad"]
COLORS = ["#b82731", "#59813b", "#6745a0", "#d97b05", "#637082", "#00828c", "#2964ac"]


def read_predictions(results, mode):
    with (results / f"{mode}_predictions.csv").open(newline="", encoding="utf-8") as stream:
        return {row["id"]: row for row in csv.DictReader(stream)}


def summarize(results):
    metrics = {mode: json.loads((results / f"{mode}_metrics.json").read_text(encoding="utf-8")) for mode in MODES}
    tables = {mode: read_predictions(results, mode) for mode in MODES}
    speech = tables["speech"]
    if any(table.keys() != speech.keys() for table in tables.values()):
        raise ValueError("The three model results must use the same held-out clips")
    rows = []
    for clip, speech_row in speech.items():
        truth = speech_row["emotion"]
        predictions = {mode: tables[mode][clip]["prediction"] for mode in MODES}
        if any(table[clip]["emotion"] != truth for table in tables.values()):
            raise ValueError(f"Different test labels for {clip}")
        rows.append({"id": clip, "speaker": speech_row["speaker"], "transcript": speech_row["transcript"],
                     "emotion": truth, **predictions, "fusion_confidence": float(tables["fusion"][clip]["confidence"])})
    errors = sorted((row for row in rows if row["fusion"] != row["emotion"]),
                    key=lambda row: -row["fusion_confidence"])
    if len(errors) < 5:
        errors.extend(row for row in rows if row["fusion"] == row["emotion"] and row["speech"] != row["emotion"])
    if len(errors) < 5:
        errors.extend(row for row in rows if row["fusion"] == row["speech"] == row["emotion"] and row["text"] != row["emotion"])
    by_emotion = {}
    for emotion in EMOTIONS:
        subset = [row for row in rows if row["emotion"] == emotion]
        gains = sum(row["speech"] != emotion and row["fusion"] == emotion for row in subset)
        losses = sum(row["speech"] == emotion and row["fusion"] != emotion for row in subset)
        by_emotion[emotion] = {"gained": gains, "lost": losses}
    (results / "paired_analysis.json").write_text(
        json.dumps({"fusion_vs_speech": by_emotion, "failure_cases": errors[:5]}, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = ["# TESS held-out results", "", f"Test set: {len(rows)} clips. Accuracy for uniform guessing is 14.29%.", "",
             "| Modality | Accuracy | Macro F1 | Selected epoch |", "| --- | ---: | ---: | ---: |"]
    for mode, title in MODES.items():
        score = metrics[mode]
        lines.append(f"| {title} | {score['accuracy']:.2%} | {score['macro avg']['f1-score']:.4f} | {score['validation_selected_epoch']} |")
    lines.extend(["", "## Per-class recall and paired fusion changes", "",
                  "| Emotion | Speech | Text | Fusion | Fusion gains | Fusion losses |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for emotion, name in zip(EMOTIONS, NAMES):
        recalls = " | ".join(f"{metrics[mode][emotion]['recall']:.2%}" for mode in MODES)
        change = by_emotion[emotion]
        lines.append(f"| {name} | {recalls} | {change['gained']} | {change['lost']} |")
    lines.extend(["", "Each gain is a corrected speech-only error. Each loss is a correct speech-only prediction that fusion misclassifies.",
                  "", "## Failure cases", "",
                  "Cases prioritize high-confidence fusion errors, followed by speech-only and text-only errors as needed.", "",
                  "| Clip | True emotion | Speech | Text | Fusion | Fusion confidence |",
                  "| --- | --- | --- | --- | --- | ---: |"])
    for row in errors[:5]:
        lines.append(f"| {row['id']} | {row['emotion']} | {row['speech']} | {row['text']} | {row['fusion']} | {row['fusion_confidence']:.1%} |")
    (results / "accuracy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(results):
    plots = results / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    comparison, panels = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    for (mode, title), axis, panel in zip(MODES.items(), axes, panels):
        with np.load(results / f"{mode}_representations.npz") as data:
            embedding, labels, confusion = data["embedding"], data["labels"], data["confusion"]
        pca = PCA(n_components=2)
        coordinates = pca.fit_transform(embedding)
        if mode == "text":
            unique = np.unique(coordinates.round(decimals=5), axis=0)
            axis.scatter(*unique.T, color="#495666", s=17, alpha=0.7, edgecolors="none")
            axis.set_title(f"{title}\nEach transcript includes all labels")
        else:
            for index, (name, color) in enumerate(zip(NAMES, COLORS)):
                subset = coordinates[labels == index]
                axis.scatter(*subset.T, label=name, color=color, s=13, alpha=0.32, edgecolors="none")
            axis.set_title(title)
        axis.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.0%})")
        axis.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.0%})")
        image = panel.imshow(confusion, vmin=0, vmax=confusion.sum(axis=1).max(), cmap="Blues")
        for row, column in np.ndindex(confusion.shape):
            count = confusion[row, column]
            panel.text(column, row, str(count), ha="center", va="center",
                       color="white" if count > 34 else "#172536", fontsize=9)
        panel.set_title(title)
        panel.set_xticks(range(len(NAMES)), NAMES, rotation=45, ha="right")
        panel.set_yticks(range(len(NAMES)), NAMES if mode == "speech" else [])
        panel.set_xlabel("Predicted emotion")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=7)
    figure.suptitle("Learned representations on held-out words")
    panels[0].set_ylabel("True emotion")
    comparison.colorbar(image, ax=list(panels), shrink=0.75, label="Test clips")
    comparison.suptitle("Emotion confusion matrices")
    figure.savefig(plots / "representations.png", dpi=170)
    comparison.savefig(plots / "confusion.png", dpi=170)
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "Results")
    args = parser.parse_args()
    summarize(args.results)
    plot(args.results)
