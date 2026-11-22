"""Measure agreement between our ARPAbet alignments and two public sets of TextGrids."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import json
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from compare_models import labeled, read_tiers


HERE = Path(__file__).resolve().parent
SOURCES = {
    "KompallyNikshitha": ("KompallyNikshitha/Montreal-Forced-Aligner",
                          "518e9084cfac363660a1764da82c10eeda1cab3a", "output"),
    "atharva753": ("atharva753/mfa-forced-alignment-pipeline",
                   "1f4b282327daf5e84067b6a55fe11165cc976ea9", "mfa_output_english_arpa"),
}


def compare_clip(path, source, work):
    repository, revision, folder = source
    url = f"https://raw.githubusercontent.com/{repository}/{revision}/{folder}/{path.name}"
    with urlopen(url, timeout=25) as response:
        work.write_bytes(response.read())
    local, _ = read_tiers(path)
    remote, phones = read_tiers(work)
    local, remote = labeled(local), labeled(remote)
    phones = labeled(phones)
    matcher = SequenceMatcher(None, [r[2].casefold() for r in local],
                              [r[2].casefold() for r in remote], autojunk=False)
    pairs = [(local[i + offset], remote[j + offset])
             for i, j, size in matcher.get_matching_blocks() for offset in range(size)]
    shifts = [abs(a[side] - b[side]) * 1000 for a, b in pairs for side in (0, 1)]
    return {
        "clip": path.stem, "source": url, "local_words": len(local), "external_words": len(remote),
        "matched_words": len(pairs), "long_phones": sum(end - start > 0.5 for start, end, _ in phones),
        "unknown_phones": sum(label.casefold() in ("spn", "<unk>") for _, _, label in phones),
    }, shifts


def compare(directory, output):
    paths = sorted(directory.rglob("*.TextGrid"))
    if not paths:
        raise ValueError(f"No TextGrids found in {directory}")
    results = {}
    with TemporaryDirectory() as temp, ThreadPoolExecutor(max_workers=4) as pool:
        for name, source in SOURCES.items():
            jobs = [pool.submit(compare_clip, path, source, Path(temp) / path.name) for path in paths]
            reports, shifts = zip(*(job.result() for job in jobs))
            shifts = [shift for clip in shifts for shift in clip]
            results[name] = {
                "revision": source[1], "clips": reports, "boundaries_compared": len(shifts),
                "median_shift_ms": round(median(shifts), 2), "mean_shift_ms": round(mean(shifts), 2),
                "max_shift_ms": round(max(shifts), 2), "within_20_ms": sum(shift <= 20.5 for shift in shifts),
                "within_50_ms": sum(shift <= 50.5 for shift in shifts),
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=Path, default=HERE / "TextGrids_us_arpa")
    parser.add_argument("--output", type=Path, default=HERE / "Results/comparison/external_agreement.json")
    args = parser.parse_args()
    compare(args.grids, args.output)
