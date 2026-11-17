"""Build the shared TESS splits and acoustic features from the extracted archive."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from math import gcd
from pathlib import Path
import random
import re

import numpy as np
from scipy.signal import resample_poly, stft
import soundfile as sf


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad")
ALPHABET = " abcdefghijklmnopqrstuvwxyz'-"
NAME = re.compile(r"(OAF|YAF)_([a-z'-]+)_(angry|disgust|fear|happy|neutral|ps|sad)", re.I)
RATE = 16000


@dataclass(frozen=True)
class Recording:
    path: Path
    speaker: str
    word: str
    emotion: str


def discover(source):
    recordings = {}
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() != ".wav":
            continue
        # Both copies in the Kaggle archive omit the F in this one filename.
        stem = "OAF_bite_neutral" if path.stem == "OA_bite_neutral" else path.stem
        match = NAME.fullmatch(stem)
        if not match:
            raise ValueError(f"Unexpected TESS WAV filename: {path.name}")
        speaker, word, label = match.groups()
        label = "pleasant_surprise" if label.lower() == "ps" else label.lower()
        key = (speaker.upper(), word.lower(), label)
        if key in recordings:
            if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(recordings[key].path.read_bytes()).digest():
                raise ValueError(f"Conflicting TESS files for {key}: {path}")
        else:
            recordings[key] = Recording(path, *key)
    words = {sample.word for sample in recordings.values()}
    expected = {(speaker, word, label) for speaker in ("OAF", "YAF") for word in words for label in EMOTIONS}
    if len(words) != 200 or set(recordings) != expected:
        raise ValueError(f"Expected 200 words x 2 speakers x 7 emotions, found {len(recordings)} unique recordings")
    return list(recordings.values()), sorted(words)


def mel_filterbank():
    hz = np.fft.rfftfreq(512, 1 / RATE)
    edges = 700 * np.expm1(np.linspace(np.log1p(60 / 700), np.log1p(7600 / 700), 66))
    left, middle, right = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0, np.minimum((hz - left) / (middle - left), (right - hz) / (right - middle)))


MEL = mel_filterbank()


def log_mel(path):
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError(f"Empty or non-finite signal: {path}")
    samples = samples.mean(axis=1)
    samples -= samples.mean()
    peak = np.max(np.abs(samples))
    if peak == 0 or not np.isfinite(peak):
        raise ValueError(f"Empty or non-finite signal: {path}")
    samples /= peak
    divisor = gcd(rate, RATE)
    samples = resample_poly(samples, RATE // divisor, rate // divisor)
    # Use a short RMS envelope to trim quiet edges while preserving internal pauses.
    window = 320
    envelope = np.sqrt(np.mean(np.pad(samples, (0, -len(samples) % window)).reshape(-1, window) ** 2, axis=1))
    audible = np.flatnonzero(envelope >= envelope.max() * 0.025)
    start = max(0, audible[0] - 2) * window
    end = (audible[-1] + 3) * window
    samples = samples[start:end]
    if len(samples) < 400:
        samples = np.pad(samples, (0, 400 - len(samples)))
    _, _, spectrum = stft(samples, fs=RATE, nperseg=400, noverlap=240, nfft=512)
    return np.log(np.maximum(MEL @ (np.abs(spectrum) ** 2), 1e-8)).T.astype(np.float32)


def encode_text(text):
    """Use the same normalization and character vocabulary for every entry point."""
    cleaned = " ".join(text.lower().split())
    if not cleaned:
        raise ValueError("Text must contain at least one non-whitespace character")
    tokens = {character: index + 2 for index, character in enumerate(ALPHABET)}
    return [tokens.get(character, 1) for character in cleaned]


def prepare(source, destination, seed):
    recordings, words = discover(source)
    random.Random(seed).shuffle(words)
    assignments = {word: split for split, group in (
        ("train", words[:140]), ("validation", words[140:170]), ("test", words[170:])
    ) for word in group}
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        subset = [sample for sample in recordings if assignments[sample.word] == split]
        features = [log_mel(sample.path) for sample in subset]
        lengths = np.array([len(feature) for feature in features])
        metadata = [
            {"id": sample.path.stem, "speaker": sample.speaker, "word": sample.word,
             "transcript": f"say the word {sample.word}", "emotion": sample.emotion}
            for sample in subset
        ]
        np.savez_compressed(destination / f"{split}.npz", frames=np.concatenate(features), lengths=lengths,
                            labels=[EMOTIONS.index(sample.emotion) for sample in subset])
        (destination / f"{split}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"{split}: {len(subset)} clips, {len({sample.word for sample in subset})} words", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=DATA)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed)
