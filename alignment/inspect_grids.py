"""Inspect MFA TextGrids with the Praat engine and plot a sample alignment."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import spectrogram


HERE = Path(__file__).resolve().parent


def intervals(grid, tier):
    if call(grid, "Get tier name", tier) not in ("words", "phones"):
        raise ValueError(f"Unexpected tier {tier} in {grid.name}")
    count = call(grid, "Get number of intervals", tier)
    entries = [(call(grid, "Get start time of interval", tier, i),
             call(grid, "Get end time of interval", tier, i),
             call(grid, "Get label of interval", tier, i))
            for i in range(1, count + 1)]
    previous = call(grid, "Get start time")
    for start, end, label in entries:
        if end <= start or abs(start - previous) > 1e-6:
            raise ValueError(f"Invalid interval in tier {tier}: {(start, end, label)}")
        previous = end
    if abs(previous - call(grid, "Get end time")) > 1e-6:
        raise ValueError(f"Incomplete tier {tier}")
    return entries


def plot_example(sound, words, phones, clip, output):
    samples = sound.values[0]
    sample_rate = sound.sampling_frequency
    times = sound.xs()
    start, end = words[0][0] - 0.10, words[-1][1] + 0.10
    start, end = max(start, 0), min(end, sound.duration)
    window = (times >= start) & (times <= end)
    frequency, frames, power = spectrogram(samples[window], fs=sample_rate, nperseg=256, noverlap=208)
    power_db = 10 * np.log10(np.maximum(power, 1e-13))
    vmax = power_db.max()
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 5.1), layout="constrained",
                                gridspec_kw={"height_ratios": [1.4, 2, 0.52, 0.52]})
    axes[0].plot(times[window], samples[window], color="#244c78", linewidth=0.6)
    axes[0].set_ylabel("Amplitude")
    axes[1].pcolormesh(frames + times[window][0], frequency / 1000, power_db, shading="auto",
                       cmap="magma", vmin=vmax - 65, vmax=vmax)
    axes[1].set_ylim(0, 7)
    axes[1].set_ylabel("Frequency (kHz)")
    for axis, tier, label in zip(axes[2:], [words, phones], ["Words", "Phones"]):
        axis.set_ylim(0, 1)
        axis.set_yticks([0.5], [label])
        axis.tick_params(axis="y", length=0)
        for left, right, text in tier:
            axis.axvspan(left, right, facecolor="#edf3f9", edgecolor="#58789c")
            axis.text((left + right) / 2, 0.5, text, ha="center", va="center", fontsize=9)
            for acoustic_axis in axes[:2]:
                acoustic_axis.axvline(left, color="white" if acoustic_axis is axes[1] else "#5c7592",
                                     alpha=0.4, linewidth=0.55)
    axes[-1].set_xlim(start, end)
    axes[-1].set_xlabel("Time (seconds)")
    figure.suptitle(f"MFA word and phone alignment | {clip}")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def inspect(corpus, grids, output, sample):
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(grids.rglob("*.TextGrid"))
    if not paths:
        raise ValueError(f"No TextGrids found in {grids}")
    expected = {path.relative_to(corpus).with_suffix(".TextGrid") for path in corpus.rglob("*.wav")}
    actual = {path.relative_to(grids) for path in paths}
    if expected != actual:
        raise ValueError(f"Corpus/TextGrid mismatch. Missing: {expected - actual}. Extra: {actual - expected}.")
    if sample not in {path.stem for path in paths}:
        raise ValueError(f"Sample {sample!r} was not found in {grids}")
    overview = []
    review = []
    with (output / "intervals.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["clip", "tier", "start_seconds", "end_seconds", "label"])
        for path in paths:
            wav = corpus / path.relative_to(grids).with_suffix(".wav")
            sound = parselmouth.Sound(str(wav))
            grid = parselmouth.Data.read(str(path))
            if (call(grid, "Get number of tiers") != 2 or call(grid, "Get tier name", 1) != "words"
                    or call(grid, "Get tier name", 2) != "phones"):
                raise ValueError(f"Expected word and phone tiers in {path}")
            if abs(call(grid, "Get end time") - sound.duration) > 0.001:
                raise ValueError(f"Duration mismatch for {path.name}")
            words, phones = (intervals(grid, tier) for tier in (1, 2))
            words = [row for row in words if row[2].strip()]
            phones = [row for row in phones if row[2].strip()]
            for start, end, label in phones:
                if end - start > 0.5:
                    review.append((path.stem, f"{start:.6f}", f"{end:.6f}", label, "phone exceeds 0.5 seconds"))
            if not words or not phones:
                raise ValueError(f"Missing labeled word or phone intervals in {path}")
            overview.append((path.stem, f"{sound.duration:.3f}", len(words), len(phones),
                             sum(label in ("spn", "<unk>") for _, _, label in phones)))
            for name, tier in [("words", words), ("phones", phones)]:
                writer.writerows((path.stem, name, f"{start:.6f}", f"{end:.6f}", label) for start, end, label in tier)
            if path.stem == sample:
                plot_example(sound, words, phones, path.stem, output / "sample_alignment.png")
    with (output / "overview.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["clip", "duration_seconds", "words", "phones", "unknown_phones"])
        writer.writerows(overview)
    with (output / "review_flags.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["clip", "start_seconds", "end_seconds", "phone", "reason"])
        writer.writerows(review)
    print(f"Inspected {len(paths)} TextGrids with Praat {parselmouth.PRAAT_VERSION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=HERE / "corpus")
    parser.add_argument("--grids", type=Path, default=HERE / "TextGrids")
    parser.add_argument("--output", type=Path, default=HERE / "Results")
    parser.add_argument("--sample", default="ISLE_SESS0131_BLOCKD02_01_sprt1")
    args = parser.parse_args()
    inspect(args.corpus, args.grids, args.output, args.sample)
