"""Compare repo-inspired baselines on the assignment's frozen TESS split."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
from scipy.fft import dct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
import torch

from emotion.data_prep import DATA, EMOTIONS, HERE
from emotion.experiment import TessSplit, evaluate, load_model, split_fingerprints, train


def prepare_cepstra(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        source_metadata = source / f"{split}.json"
        (destination / source_metadata.name).write_bytes(source_metadata.read_bytes())
        with np.load(source / f"{split}.npz") as data:
            # Keep the model input shape fixed so only the feature representation changes.
            frames = dct(data["frames"], type=2, axis=1, norm="ortho")
            frames[:, 40:] = 0
            np.savez_compressed(destination / f"{split}.npz", frames=frames, lengths=data["lengths"],
                                labels=data["labels"])


def probabilities(mode, checkpoint_dir, directory, split, args, device):
    dataset = TessSplit(directory, split, mode)
    model, checkpoint = load_model(mode, checkpoint_dir / f"{mode}.pt", device)
    return evaluate(model, dataset, args, device, checkpoint["stats"])[1]


def compare_seed(args, device, cepstra, seed):
    checkpoints = args.work / f"seed_{seed}"
    original_checkpoints = args.checkpoints if seed == 42 else args.seed_study / f"seed_{seed}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train("speech", argparse.Namespace(data=cepstra, checkpoints=checkpoints, batch_size=64,
                                      seed=seed, epochs=30, patience=8), device)

    args.batch_size = 64
    datasets = {split: TessSplit(args.data, split, "text") for split in ("train", "validation", "test")}
    transcripts = {split: [row["transcript"] for row in ds.metadata] for split, ds in datasets.items()}
    labels = {split: ds.labels.numpy() for split, ds in datasets.items()}
    vectorizer = TfidfVectorizer()
    text_model = LogisticRegression(random_state=seed)
    text_model.fit(vectorizer.fit_transform(transcripts["train"]), labels["train"])
    assert np.array_equal(text_model.classes_, np.arange(len(EMOTIONS)))
    text_scores = {split: text_model.predict_proba(vectorizer.transform(transcripts[split]))
                   for split in ("validation", "test")}
    original = {"speech": "log_mel_bigru", "text": "character_bigru", "fusion": "representation_fusion"}
    predictions = {name: probabilities(mode, original_checkpoints, args.data, "test", args, device).argmax(axis=1)
                   for mode, name in original.items()}
    predictions["tfidf_logistic"] = text_scores["test"].argmax(axis=1)
    for name, folder, data in (("log_mel", original_checkpoints, args.data),
                               ("mfcc", checkpoints, cepstra)):
        speech_scores = {split: probabilities("speech", folder, data, split, args, device)
                         for split in ("validation", "test")}
        predictions[f"{name}_bigru"] = speech_scores["test"].argmax(axis=1)
        inputs = {split: np.concatenate((speech_scores[split], text_scores[split]), axis=1)
                  for split in ("validation", "test")}
        fusion = MLPClassifier(hidden_layer_sizes=(32,), solver="lbfgs", max_iter=1000, random_state=seed)
        fusion.fit(inputs["validation"], labels["validation"])
        predictions[f"{name}_score_fusion"] = fusion.predict(inputs["test"])

    truth = labels["test"]
    speech = predictions["log_mel_bigru"] == truth
    scores = {}
    for name, predicted in predictions.items():
        correct = predicted == truth
        scores[name] = {"correct": int(correct.sum()), "accuracy": float(accuracy_score(truth, predicted)),
                        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
                        "corrected_vs_log_mel": int((correct & ~speech).sum()),
                        "lost_vs_log_mel": int((~correct & speech).sum())}
    print(f"Seed {seed} complete", flush=True)
    return {"test_clips": len(truth), "scores": scores}


def compare(args):
    device = torch.device("cpu")
    torch.set_num_threads(4)
    split_fingerprints(args.data)
    cepstra = args.work / "cepstra"
    prepare_cepstra(args.data, cepstra)
    results = {seed: compare_seed(args, device, cepstra, seed) for seed in args.seeds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seeds": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--checkpoints", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--work", type=Path, default=HERE.parent / ".local/baseline-comparison")
    parser.add_argument("--output", type=Path, default=HERE / "Results/baseline_comparison.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    parser.add_argument("--seed-study", type=Path, default=HERE.parent / ".local/seed-study")
    compare(parser.parse_args())
