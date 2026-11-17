"""Prepare the supplied corpus and run Montreal Forced Aligner."""

import argparse
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MODEL = "english_mfa"


def normalize(transcript):
    transcript = re.sub(r"\bS\.J\.C\.(?='s)", "S J C", transcript)
    transcript = re.sub(r"\bWBUR\b", "W B U R", transcript)
    transcript = re.sub(r"-\s+", " ", transcript)
    return " ".join(transcript.lower().split())


def by_stem(directory, suffix):
    files = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() != suffix:
            continue
        stem = path.stem.casefold()
        if stem in files:
            raise ValueError(f"Duplicate recording stem: {files[stem]} and {path}")
        files[stem] = path
    return files


def prepare(source, corpus):
    wavs = by_stem(source / "wav", ".wav")
    transcripts = by_stem(source / "transcripts", ".txt")
    if not wavs or wavs.keys() != transcripts.keys():
        raise ValueError(f"Unpaired corpus files: {sorted(wavs.keys() ^ transcripts.keys())}")
    files = []
    for stem, wav in sorted(wavs.items()):
        speaker = "F2BJ" if stem.startswith("f2bj") else wav.stem.split("_")[1]
        files.append((wav, transcripts[stem], corpus / speaker / wav.name))
    expected = {path for _, _, target in files for path in (target, target.with_suffix(".lab"))}
    existing = {path for path in corpus.rglob("*") if path.suffix.lower() in (".wav", ".lab")}
    if existing - expected:
        raise ValueError(f"Unexpected prepared files in {corpus}. Use a fresh --corpus directory.")
    for wav, transcript_path, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        transcript = normalize(transcript_path.read_text(encoding="utf-8-sig"))
        if not transcript:
            raise ValueError(f"Empty transcript: {transcript_path}")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(target)],
            check=True,
        )
        target.with_suffix(".lab").write_text(transcript + "\n", encoding="utf-8")
    print(f"Prepared {len(wavs)} recordings in {corpus}", flush=True)


def extend_dictionary(dictionary, additions, destination):
    base = dictionary.read_text(encoding="utf-8").splitlines()
    rows = [row.split("\t") for row in base if row.strip()]
    if any(len(row) != 2 for row in rows):
        raise ValueError("Use a plain two-column MFA dictionary with --lexicon-additions")
    words = {word for word, _ in rows}
    inventory = {phone for _, pronunciation in rows for phone in pronunciation.split()}
    for row in additions.read_text(encoding="utf-8").splitlines():
        word, pronunciation = row.split("\t")
        unknown = set(pronunciation.split()) - inventory
        if unknown:
            raise ValueError(f"Unsupported phones for {word}: {sorted(unknown)}")
        if word not in words:
            base.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(base) + "\n", encoding="utf-8")
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "Assignment 2a")
    parser.add_argument("--corpus", type=Path, default=ROOT / "alignment/corpus")
    parser.add_argument("--output", type=Path, default=ROOT / "alignment/TextGrids")
    parser.add_argument("--dictionary", default=MODEL)
    parser.add_argument("--lexicon-additions", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    prepare(args.source, args.corpus)
    if args.prepare_only:
        return
    os.environ.setdefault("MFA_ROOT_DIR", str(ROOT / ".local/mfa-work"))
    dictionary = args.dictionary
    if args.lexicon_additions:
        dictionary = extend_dictionary(Path(dictionary), args.lexicon_additions, args.corpus.parent / "extended.dict")
    for kind, model in [("dictionary", dictionary), ("acoustic", MODEL)]:
        if model == MODEL:
            subprocess.run(["mfa", "model", "download", kind, model], check=True)
    if args.validate:
        subprocess.run(
            ["mfa", "validate", str(args.corpus), dictionary, MODEL, "--skip_acoustics", "--clean", "-j", "2"],
            check=True,
        )
    subprocess.run(
        ["mfa", "align", str(args.corpus), dictionary, MODEL, str(args.output),
         "--clean", "--overwrite", "-j", "2"],
        check=True,
    )
    expected = {p.relative_to(args.corpus).with_suffix(".TextGrid")
                for p in args.corpus.rglob("*.wav")}
    actual = {p.relative_to(args.output) for p in args.output.rglob("*.TextGrid")}
    if expected != actual:
        raise RuntimeError(f"TextGrid inventory mismatch. Missing: {expected - actual}. Extra: {actual - expected}.")


if __name__ == "__main__":
    main()
