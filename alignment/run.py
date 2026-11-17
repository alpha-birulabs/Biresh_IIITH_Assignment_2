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
    """Append pronunciations for missing words, keeping to the dictionary's phone inventory."""
    base = dictionary.read_text(encoding="utf-8").splitlines()
    # The pronunciation is the last column. Probability columns may sit between it and the word.
    rows = [row.split("\t") for row in base if row.strip()]
    if any(len(row) < 2 for row in rows):
        raise ValueError(f"Expected tab-separated word and pronunciation columns in {dictionary}")
    words = {row[0] for row in rows}
    inventory = {phone for row in rows for phone in row[-1].split()}
    added = []
    for row in additions.read_text(encoding="utf-8").splitlines():
        if not row.strip():
            continue
        word, pronunciation = row.split("\t")
        unknown = set(pronunciation.split()) - inventory
        if unknown:
            raise ValueError(f"Unsupported phones for {word}: {sorted(unknown)}")
        if word not in words:
            added.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(base + added) + "\n", encoding="utf-8")
    print(f"Added {len(added)} pronunciations to {destination}", flush=True)
    return destination


def pretrained_dictionary(name):
    """Return the downloaded file behind a pretrained dictionary name."""
    subprocess.run(["mfa", "model", "download", "dictionary", name], check=True)
    path = Path(os.environ["MFA_ROOT_DIR"]) / "pretrained_models/dictionary" / f"{name}.dict"
    if not path.is_file():
        raise FileNotFoundError(f"MFA did not save {name} at {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "Assignment 2a")
    parser.add_argument("--corpus", type=Path, default=ROOT / "alignment/corpus")
    parser.add_argument("--output", type=Path, default=ROOT / "alignment/TextGrids")
    parser.add_argument("--acoustic", default=MODEL, help="pretrained MFA acoustic model name")
    parser.add_argument("--dictionary", help="dictionary name or file (default: same name as --acoustic)")
    parser.add_argument("--lexicon-additions", type=Path, help="TSV of hand-written pronunciations to add")
    parser.add_argument("--g2p", help="pretrained MFA G2P model that transcribes out-of-vocabulary words")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.lexicon_additions and args.g2p:
        parser.error("Use either --lexicon-additions or --g2p")
    prepare(args.source, args.corpus)
    if args.prepare_only:
        return
    os.environ.setdefault("MFA_ROOT_DIR", str(ROOT / ".local/mfa-work"))
    dictionary = args.dictionary or args.acoustic
    extended = args.corpus.parent / f"{args.output.name}.extended.dict"
    if args.lexicon_additions:
        dictionary = extend_dictionary(Path(dictionary), args.lexicon_additions, extended)
    elif args.g2p:
        base = Path(dictionary) if Path(dictionary).is_file() else pretrained_dictionary(dictionary)
        subprocess.run(["mfa", "model", "download", "g2p", args.g2p], check=True)
        generated = ROOT / f"alignment/pronunciations_g2p_{args.g2p}.tsv"
        # One pronunciation per out-of-vocabulary word keeps the generated lexicon reviewable.
        subprocess.run(
            ["mfa", "g2p", str(args.corpus), args.g2p, str(generated), "--dictionary_path", str(base),
             "--num_pronunciations", "1", "--sorted", "--clean", "-j", "2"],
            check=True,
        )
        dictionary = extend_dictionary(base, generated, extended)
    elif not Path(dictionary).is_file():
        subprocess.run(["mfa", "model", "download", "dictionary", dictionary], check=True)
    dictionary = str(dictionary)
    subprocess.run(["mfa", "model", "download", "acoustic", args.acoustic], check=True)
    if args.validate:
        subprocess.run(
            ["mfa", "validate", str(args.corpus), dictionary, args.acoustic, "--skip_acoustics", "--clean", "-j", "2"],
            check=True,
        )
    subprocess.run(
        ["mfa", "align", str(args.corpus), dictionary, args.acoustic, str(args.output),
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
