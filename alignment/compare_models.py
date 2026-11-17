"""Compare the word boundaries that several MFA runs give for the same corpus."""

import argparse
import csv
from itertools import combinations
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from scipy.signal import spectrogram

from inspect_grids import intervals


HERE = Path(__file__).resolve().parent
# Run name -> TextGrid directory. Adjacent runs differ in one component only.
RUNS = {"mfa_generic": HERE / "TextGrids", "mfa_us": HERE / "TextGrids_us_mfa", "arpa_us": HERE / "TextGrids_us_arpa"}
FIGURES = [("ISLE_SESS0131_BLOCKD02_01_sprt1", None), ("F2BJRLP1", (19.2, 21.5))]
STYLES = [("#edf3f9", "#58789c"), ("#eef6ec", "#5d8a54"), ("#fbf0e2", "#b0742a"), ("#f3edf8", "#7a5a9c")]


def read_tiers(path):
    grid = parselmouth.Data.read(str(path))
    return intervals(grid, 1), intervals(grid, 2)


def labeled(tier):
    return [row for row in tier if row[2].strip()]


def load(runs):
    """Return {clip: (speaker, {run: (words, phones)})} after checking that the runs are comparable."""
    inventories = {name: {p.relative_to(d) for p in d.rglob("*.TextGrid")} for name, d in runs.items()}
    reference = next(iter(inventories.values()))
    if not reference or any(inventory != reference for inventory in inventories.values()):
        raise ValueError(f"The runs must hold TextGrids for the same recordings: {list(runs.values())}")
    clips = {}
    for relative in sorted(reference):
        tiers = {name: read_tiers(directory / relative) for name, directory in runs.items()}
        sequences = {tuple(word[2] for word in labeled(words)) for words, _ in tiers.values()}
        if len(sequences) != 1:
            raise ValueError(f"The runs disagree on the word sequence of {relative.stem}")
        clips[relative.stem] = (relative.parent.name, tiers)
    return clips


def agreement(rows, first, second):
    lines = [f"### {first} vs {second}", "",
             "| Speaker | Word tokens | Boundary | Median (ms) | Mean (ms) | Within 20 ms | Within 50 ms | Largest (ms) |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for speaker in sorted({row["speaker"] for row in rows}) + ["All"]:
        subset = [row for row in rows if speaker in (row["speaker"], "All")]
        for side in ("start", "end"):
            shifts = [abs(row[second][side] - row[first][side]) for row in subset]
            # Boundaries sit on a 10 ms frame grid, so allow for float rounding at the thresholds.
            lines.append(f"| {speaker} | {len(subset)} | {side} | {1000 * median(shifts):.0f} | {1000 * mean(shifts):.0f} | "
                         f"{mean(shift <= 0.0205 for shift in shifts):.0%} | {mean(shift <= 0.0505 for shift in shifts):.0%} | "
                         f"{1000 * max(shifts):.0f} |")
    lines.extend(["", f"Largest word-end shifts ({second} minus {first}):", "",
                  f"| Recording | Word | {first} end (s) | {second} end (s) | Shift (ms) |", "| --- | --- | ---: | ---: | ---: |"])
    for row in sorted(rows, key=lambda row: -abs(row[second]["end"] - row[first]["end"]))[:6]:
        lines.append(f"| {row['clip']} | {row['word']} | {row[first]['end']:.3f} | {row[second]['end']:.3f} | "
                     f"{1000 * (row[second]['end'] - row[first]['end']):+.0f} |")
    return lines + [""]


def compare(clips, names, output):
    rows = []
    for clip, (speaker, tiers) in clips.items():
        for token in zip(*(labeled(tiers[name][0]) for name in names)):
            rows.append({"speaker": speaker, "clip": clip, "word": token[0][2],
                         **{name: {"start": start, "end": end} for name, (start, end, _) in zip(names, token)}})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "word_boundaries.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["speaker", "clip", "word"] + [f"{name}_{side}" for name in names for side in ("start", "end")])
        for row in rows:
            writer.writerow([row["speaker"], row["clip"], row["word"]]
                            + [f"{row[name][side]:.3f}" for name in names for side in ("start", "end")])
    lines = ["# Word-boundary agreement between MFA runs", "",
             "Every run aligns the same six recordings. A shift is the absolute difference between two runs for the "
             "same word token. The corpus has no reference boundaries, so a shift shows disagreement, not error.", "",
             "## Per recording", "",
             "| Recording | Words | " + " | ".join(f"Phones ({name}) | Pauses ({name}) | Longest phone s ({name})" for name in names) + " |",
             "| --- | ---: |" + " ---: | ---: | ---: |" * len(names)]
    for clip, (_, tiers) in clips.items():
        cells = []
        for name in names:
            words, phones = tiers[name]
            cells += [str(len(labeled(phones))), str(len(words) - len(labeled(words))),
                      f"{max(end - start for start, end, _ in labeled(phones)):.2f}"]
        lines.append(f"| {clip} | {len(labeled(tiers[names[0]][0]))} | " + " | ".join(cells) + " |")
    lines.extend(["", "Pauses are unlabeled word intervals. A longest phone above 0.5 s marks a boundary to review.", "",
                  "## Pairwise agreement", ""])
    for first, second in combinations(names, 2):
        lines.extend(agreement(rows, first, second))
    (output / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Compared {len(rows)} word tokens in {len(clips)} recordings across {len(names)} runs")
    return rows


def plot(corpus, clips, names, clip, window, destination):
    speaker, tiers = clips[clip]
    sound = parselmouth.Sound(str(corpus / speaker / f"{clip}.wav"))
    start, end = window or (0, sound.duration)
    chosen = (sound.xs() >= start) & (sound.xs() <= end)
    samples, times = sound.values[0][chosen], sound.xs()[chosen]
    frequency, frames, power = spectrogram(samples, fs=sound.sampling_frequency, nperseg=256, noverlap=208)
    power_db = 10 * np.log10(np.maximum(power, 1e-13))
    figure, axes = plt.subplots(2 + 2 * len(names), 1, sharex=True, figsize=(11, 4.4 + 1.1 * len(names)),
                                layout="constrained",
                                gridspec_kw={"height_ratios": [1.1, 1.8] + [0.5] * (2 * len(names))})
    axes[0].plot(times, samples, color="#244c78", linewidth=0.6)
    axes[0].set_ylabel("Amplitude")
    axes[1].pcolormesh(frames + times[0], frequency / 1000, power_db, shading="auto", cmap="magma",
                       vmin=power_db.max() - 65, vmax=power_db.max())
    axes[1].set_ylim(0, 7)
    axes[1].set_ylabel("kHz")
    for index, name in enumerate(names):
        face, edge = STYLES[index % len(STYLES)]
        for axis, tier, kind in zip(axes[2 + 2 * index:], tiers[name], ("words", "phones")):
            axis.set_ylim(0, 1)
            axis.set_yticks([0.5], [f"{name}\n{kind}"], fontsize=8)
            axis.tick_params(axis="y", length=0)
            for left, right, text in labeled(tier):
                if right <= start or left >= end:
                    continue
                axis.axvspan(left, right, facecolor=face, edgecolor=edge)
                axis.text((max(left, start) + min(right, end)) / 2, 0.5, text, ha="center", va="center",
                          fontsize=7.5, clip_on=True)
    axes[-1].set_xlim(start, end)
    axes[-1].set_xlabel("Time (seconds)")
    figure.suptitle(f"MFA runs on the same recording | {clip}")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def parse_run(text):
    name, _, directory = text.partition("=")
    if not name or not directory:
        raise argparse.ArgumentTypeError("Use NAME=TEXTGRID_DIRECTORY")
    return name, Path(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=HERE / "corpus")
    parser.add_argument("--run", type=parse_run, action="append", metavar="NAME=DIRECTORY",
                        help="a run to compare; repeat it (default: the three documented runs)")
    parser.add_argument("--output", type=Path, default=HERE / "Results/comparison")
    args = parser.parse_args()
    runs = dict(args.run) if args.run else RUNS
    if len(runs) < 2:
        parser.error("Compare at least two runs")
    clips = load(runs)
    compare(clips, list(runs), args.output)
    for clip, window in FIGURES:
        if clip in clips:
            suffix = f"_{window[0]:g}-{window[1]:g}s" if window else ""
            plot(args.corpus, clips, list(runs), clip, window, args.output / f"{clip}{suffix}.png")
