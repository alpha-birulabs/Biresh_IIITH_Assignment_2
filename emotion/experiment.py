"""Train and evaluate each modality on the same word-disjoint TESS split."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from emotion.data_prep import DATA, EMOTIONS, HERE, encode_text
from emotion.models.networks import EmotionClassifier


class TessSplit(Dataset):
    def __init__(self, directory, split, mode):
        self.metadata = json.loads((directory / f"{split}.json").read_text(encoding="utf-8"))
        if not self.metadata:
            raise ValueError(f"Empty {split} split")
        self.labels = torch.tensor([EMOTIONS.index(item["emotion"]) for item in self.metadata])
        self.frames = None
        self.tokens = None
        if mode in ("speech", "fusion"):
            with np.load(directory / f"{split}.npz") as data:
                self.frames = torch.from_numpy(data["frames"])
                lengths = data["lengths"]
                if (lengths.shape != (len(self.metadata),) or not np.issubdtype(lengths.dtype, np.integer)
                        or np.any(lengths < 1)):
                    raise ValueError(f"Invalid frame lengths in {split}")
                if (self.frames.ndim != 2 or self.frames.shape[1] != 64
                        or lengths.sum() != len(self.frames) or not torch.isfinite(self.frames).all()):
                    raise ValueError(f"Invalid acoustic frames in {split}")
                self.offsets = np.r_[0, lengths.cumsum()]
                if not np.array_equal(data["labels"], self.labels.numpy()):
                    raise ValueError(f"Cached audio and metadata disagree on {split} labels")
        if mode in ("text", "fusion"):
            self.tokens = [torch.tensor(encode_text(item["transcript"])) for item in self.metadata]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        speech = self.frames[self.offsets[index]:self.offsets[index + 1]] if self.frames is not None else None
        text = self.tokens[index] if self.tokens is not None else None
        return speech, text, self.labels[index]


def pad_with_lengths(sequences):
    if sequences[0] is None:
        return None, None
    return pad_sequence(sequences, batch_first=True), torch.tensor([len(sequence) for sequence in sequences])


def collate(batch):
    speech, text, labels = zip(*batch)
    return *pad_with_lengths(speech), *pad_with_lengths(text), torch.stack(labels)


def batches(dataset, args, device, stats, training=False):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=training, collate_fn=collate)
    for speech, speech_lengths, text, text_lengths, labels in loader:
        if speech is not None:
            mean, std = stats
            speech = (speech.to(device) - mean) / std
        if text is not None:
            text = text.to(device)
        yield (speech, speech_lengths, text, text_lengths), labels.to(device)


@torch.inference_mode()
def evaluate(model, dataset, args, device, stats):
    model.eval()
    truth, probabilities, embeddings = [], [], []
    for inputs, labels in batches(dataset, args, device, stats):
        logits, features = model(*inputs)
        truth.append(labels.cpu())
        probabilities.append(logits.softmax(dim=1).cpu())
        embeddings.append(features.cpu())
    return tuple(torch.cat(parts).numpy() for parts in (truth, probabilities, embeddings))


def split_fingerprints(directory):
    """Check the word-disjoint protocol and identify exactly which files were used."""
    words_seen = set()
    ids_seen = set()
    fingerprints = {}
    for split in ("train", "validation", "test"):
        metadata_path = directory / f"{split}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        words = {row["word"] for row in metadata}
        identifiers = {row["id"] for row in metadata}
        if not metadata or len(identifiers) != len(metadata):
            raise ValueError(f"Empty split or duplicate clip IDs in {split}")
        if words & words_seen or identifiers & ids_seen:
            raise ValueError(f"Data leakage into {split}: target words or clip IDs cross splits")
        words_seen.update(words)
        ids_seen.update(identifiers)
        for path in (metadata_path, directory / f"{split}.npz"):
            with path.open("rb") as stream:
                fingerprints[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return fingerprints


def save_checkpoint(checkpoint, destination):
    """Publish a complete checkpoint atomically, preserving the prior one on failure."""
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".pt.tmp")
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_model(mode, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("mode", mode) != mode:
        raise ValueError(f"Checkpoint modality does not match {mode}")
    if tuple(checkpoint.get("emotions", EMOTIONS)) != EMOTIONS:
        raise ValueError("Checkpoint emotion ordering does not match this implementation")
    model = EmotionClassifier(mode, checkpoint.get("fusion_style", "concat")).to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    return model, checkpoint


def train(mode, args, device):
    fingerprints = split_fingerprints(args.data)
    training = TessSplit(args.data, "train", mode)
    validation = TessSplit(args.data, "validation", mode)
    stats = None
    if training.frames is not None:
        stats = (training.frames.mean(dim=0).to(device), training.frames.std(dim=0).clamp_min(1e-5).to(device))
    fusion_style = getattr(args, "fusion_style", "concat")
    model = EmotionClassifier(mode, fusion_style).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    best_score = -1
    stale = 0
    history = []
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, labels in batches(training, args, device, stats, training=True):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(*inputs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        truth, probabilities, _ = evaluate(model, validation, args, device, stats)
        predicted = probabilities.argmax(axis=1)
        score = f1_score(truth, predicted, average="macro")
        history.append({"epoch": epoch, "validation_accuracy": float((predicted == truth).mean()),
                        "validation_macro_f1": float(score)})
        print(f"{mode} epoch {epoch:02d} | val accuracy {(predicted == truth).mean():.4f} | macro F1 {score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            stale = 0
            save_checkpoint(
                {"weights": model.state_dict(), "stats": stats, "epoch": epoch, "mode": mode, "fusion_style": fusion_style,
                 "emotions": list(EMOTIONS), "seed": args.seed, "data_fingerprints": fingerprints,
                 "settings": {"batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience,
                              "learning_rate": 0.002, "device": str(device), "torch_version": str(torch.__version__)}},
                args.checkpoints / f"{mode}.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    (args.checkpoints / f"{mode}_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def test(mode, args, device):
    fingerprints = split_fingerprints(args.data)
    dataset = TessSplit(args.data, "test", mode)
    model, checkpoint = load_model(mode, args.checkpoints / f"{mode}.pt", device)
    if checkpoint.get("data_fingerprints", fingerprints) != fingerprints:
        raise ValueError("Prepared data differs from the checkpoint's training data")
    truth, probabilities, embeddings = evaluate(model, dataset, args, device, checkpoint["stats"])
    predicted = probabilities.argmax(axis=1)
    labels = np.arange(len(EMOTIONS))
    metrics = classification_report(truth, predicted, labels=labels, target_names=EMOTIONS, output_dict=True, zero_division=0)
    metrics["validation_selected_epoch"] = checkpoint["epoch"]
    confusion = confusion_matrix(truth, predicted, labels=labels)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / f"{mode}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (args.results / f"{mode}_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [*dataset.metadata[0], "prediction", "confidence"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item, prediction, confidence in zip(dataset.metadata, predicted, probabilities.max(axis=1)):
            writer.writerow(dict(item, prediction=EMOTIONS[prediction], confidence=f"{confidence:.6f}"))
    np.savez_compressed(args.results / f"{mode}_representations.npz", embedding=embeddings, labels=truth, confusion=confusion)
    print(f"{mode} test accuracy {metrics['accuracy']:.4f} | macro F1 {metrics['macro avg']['f1-score']:.4f}", flush=True)


def main(mode, action):
    parser = argparse.ArgumentParser(description=f"{action.title()} the {mode} classifier")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--checkpoints", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    if action == "train":
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--patience", type=int, default=8)
    else:
        parser.add_argument("--results", type=Path, default=HERE / "Results")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive")
    if action == "train" and (args.epochs < 1 or args.patience < 1):
        parser.error("--epochs and --patience must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    (train if action == "train" else test)(mode, args, device)
