"""Run every alignment step: three MFA runs, Praat checks, comparison and the PDF report."""

import argparse
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
# TextGrid folder, results folder, and extra arguments for run.py.
RUNS = [
    ("TextGrids", "Results", ["--lexicon-additions", str(HERE / "pronunciations.tsv")]),
    ("TextGrids_us_mfa", "Results/us_mfa", ["--dictionary", "english_us_mfa", "--g2p", "english_us_mfa"]),
    ("TextGrids_us_arpa", "Results/us_arpa", ["--acoustic", "english_us_arpa", "--g2p", "english_us_arpa"]),
]


def step(script, *arguments):
    command = [sys.executable, str(HERE / script), *map(str, arguments)]
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE.parent / "Assignment 2a",
                        help="folder that holds the supplied wav/ and transcripts/")
    parser.add_argument("--generic-dictionary", type=Path, default=HERE.parent / ".local/english_mfa.dict",
                        help="plain english_mfa.dict used by the first run")
    parser.add_argument("--skip-alignment", action="store_true", help="reuse existing TextGrids")
    args = parser.parse_args()
    if not args.skip_alignment:
        for grids, _, extra in RUNS:
            dictionary = ["--dictionary", args.generic_dictionary] if grids == "TextGrids" else []
            step("run.py", "--source", args.source, "--validate", "--output", HERE / grids, *dictionary, *extra)
    for grids, results, _ in RUNS:
        step("inspect_grids.py", "--grids", HERE / grids, "--output", HERE / results)
    step("compare_models.py")
    step("build_report.py")
