# Part 2a · Montreal Forced Aligner

The six supplied recordings (two speakers, 97.2 seconds, 244 word tokens) are aligned with Montreal Forced Aligner 3.3.10. One script prepares the corpus, validates dictionary coverage, runs the aligner and checks the TextGrids. The two IPA runs hold the acoustic model fixed to show the effect of changing dictionaries. The ARPAbet run uses a different acoustic model and its matching dictionary.

| Run | Acoustic model | Dictionary | Out-of-vocabulary words | TextGrids |
| --- | --- | --- | --- | --- |
| `mfa_generic` | `english_mfa` v3.1.0 (IPA) | generic `english_mfa` v3.1.0, one pronunciation per word | 5 [hand-written entries](pronunciations.tsv) | [TextGrids/](TextGrids/) |
| `mfa_us` | `english_mfa` v3.1.0 (IPA) | `english_us_mfa`, US English with variants | 6 words from the [`english_us_mfa` G2P model](pronunciations_g2p_english_us_mfa.tsv) | [TextGrids_us_mfa/](TextGrids_us_mfa/) |
| `arpa_us` | `english_us_arpa` v3.0.0 (ARPAbet) | `english_us_arpa`, with variants | 4 words from the [`english_us_arpa` G2P model](pronunciations_g2p_english_us_arpa.tsv) | [TextGrids_us_arpa/](TextGrids_us_arpa/) |

- [PDF report](Alignment_Report.pdf) with the models, sample visualizations and observations
- [Word-boundary agreement between the runs](Results/comparison/summary.md) and the [per-word table](Results/comparison/word_boundaries.csv)
- Interval tables and Praat checks per run: [mfa_generic](Results/), [mfa_us](Results/us_mfa/), [arpa_us](Results/us_arpa/)

The final TextGrids are the six files in [TextGrids_us_arpa](TextGrids_us_arpa/). The other runs document the comparisons on the same six recordings.

| Check | Global English + generic dictionary | Global English + US IPA dictionary | US ARPAbet model and dictionary |
| --- | ---: | ---: | ---: |
| TextGrids that passed structural checks | 6 / 6 | 6 / 6 | 6 / 6 |
| Unknown phone intervals after dictionary additions | 0 | 0 | 0 |
| Phones longer than 0.5 s | 2 | 2 | 0 |
| End of *bait* in clip 01 | 4.125 s | 4.125 s | 2.250 s |
| End of *bed* in clip 03 | 4.500 s | 4.500 s | 2.060 s |
| *As* in F2BJRLP1 | 19.68–19.92 s | 20.35–20.57 s | 20.36–20.56 s |

The ARPAbet pair separates both quiet tails from the final consonants and places *as* after the audible pause. The listening review supports these local improvements. The dataset has no manually timed references for measuring boundary accuracy across the corpus.

### Comparison with public outputs on the same recordings

The published TextGrids from two other implementations cover the same six clips. The comparison pairs word labels in transcript order and calculates start and end differences for matched labels.

| Output | Labelled words | Unknown phone intervals | Phones longer than 0.5 s | Matched boundaries within 20 ms of ours |
| --- | ---: | ---: | ---: | ---: |
| Our ARPAbet run | 244 | 0 | 0 | Reference for this comparison |
| KompallyNikshitha | 238 | 5 | 2 | 450 / 464 |
| atharva753 | 241 | 6 | 5 | 475 / 478 |

The median shift is 0 ms in both comparisons. Our grids have fewer unknown and unusually long intervals, which supports selecting them for submission. Tokenisation differences leave some labels unmatched, and the boundary agreement measures only the matched subset. The [per-recording results](Results/comparison/external_agreement.json) record the exact source revisions. [KompallyNikshitha TextGrids](https://github.com/KompallyNikshitha/Montreal-Forced-Aligner/tree/518e9084cfac363660a1764da82c10eeda1cab3a/output) · [atharva753 TextGrids](https://github.com/atharva753/mfa-forced-alignment-pipeline/tree/1f4b282327daf5e84067b6a55fe11165cc976ea9/mfa_output_english_arpa)

Reproduce this comparison with `python alignment/compare_external.py`. The script downloads the published TextGrids into a temporary directory.

## Setup

Install [Miniforge](https://github.com/conda-forge/miniforge) and open a terminal with Conda available.

```sh
conda create -n aligner -c conda-forge "python=3.12" "montreal-forced-aligner=3.3" "kaldi=*=cpu*" "pytorch=*=cpu*" ffmpeg
conda activate aligner
python -m pip install -r alignment/requirements.txt
```

The recordings are not redistributed in this repository. Extract the supplied archive so that `Assignment 2a/wav/` and `Assignment 2a/transcripts/` sit in the repository root, or pass their parent folder with `--source`.

The first run uses the plain [English MFA v3.1.0 dictionary](https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-english_mfa-v3.1.0/english_mfa.dict). Save it as `.local/english_mfa.dict`. MFA downloads every other model by name.

## Run everything

```sh
python alignment/pipeline.py
```

This runs the three alignments, the Praat checks, the comparison and the PDF report. Add `--skip-alignment` to rebuild the tables, figures and report from existing TextGrids.

## Run one step

Data preparation groups the files by speaker, converts the audio to 16 kHz mono PCM WAV, and writes a matching UTF-8 `.lab` transcript for each WAV. It expands the written initialisms S.J.C. and WBUR into letters and repairs hyphen-space artifacts such as “seventy- six”.

```sh
python alignment/run.py --prepare-only
```

Each alignment below prepares the corpus, runs `mfa validate` for dictionary coverage, runs `mfa align`, and checks that every WAV has a TextGrid.

```sh
python alignment/run.py --validate --dictionary .local/english_mfa.dict --lexicon-additions alignment/pronunciations.tsv
python alignment/run.py --validate --dictionary english_us_mfa --g2p english_us_mfa --output alignment/TextGrids_us_mfa
python alignment/run.py --validate --acoustic english_us_arpa --g2p english_us_arpa --output alignment/TextGrids_us_arpa
```

`--acoustic` selects a pretrained acoustic model and `--dictionary` a pretrained dictionary name or a local file. `--lexicon-additions` appends hand-written pronunciations. `--g2p` finds the words missing from the dictionary, transcribes them with a pretrained G2P model, saves them as `alignment/pronunciations_g2p_<model>.tsv`, and aligns with the extended dictionary. Both options reject phones outside the dictionary's inventory.

```sh
python alignment/inspect_grids.py --grids alignment/TextGrids_us_arpa --output alignment/Results/us_arpa
python alignment/compare_models.py
python alignment/build_report.py
```

`inspect_grids.py` uses the Praat engine to check the corpus inventory, the tier names, contiguous positive-duration intervals, and durations against the WAVs. It exports an interval table, a per-clip overview, a waveform and spectrogram with both tiers, and `review_flags.csv`, which lists phones longer than 0.5 seconds. The threshold marks candidates for review and does not prove that a boundary is wrong. `compare_models.py` matches word tokens across runs and reports how far their boundaries differ.

## What the alignments show

The words tier and the phones tier are consistent in every run. Each word interval is covered exactly by its phones, and pauses appear as empty intervals in both tiers. MFA places boundaries on a 10 ms frame grid.

| Runs compared | What changes | Speaker | Median shift | Within 20 ms | Within 50 ms | Largest shift |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `mfa_generic` / `mfa_us` | dictionary | F2BJ | 0 ms | 90% | 95% | 670 ms |
| `mfa_generic` / `mfa_us` | dictionary | ISLE | 0 ms | 93% | 100% | 50 ms |
| `mfa_us` / `arpa_us` | model and dictionary | F2BJ | 0 ms | 94% | 98% | 250 ms |
| `mfa_us` / `arpa_us` | model and dictionary | ISLE | 15 ms | 63% | 70% | 2440 ms |

A shift is the absolute difference between two runs for the same word start or end. The data have no hand-labelled boundaries, so these numbers measure disagreement and not accuracy. The large shifts come from three specific errors.

**Trailing silence absorbed by the last phone.** With the `english_mfa` acoustic model, the final /t/ of *bait* in ISLE clip 01 runs from 2.050 s to the end of the file at 4.125 s, and the final /d/ of *bed* in clip 03 runs from 1.790 s to 4.500 s. Both IPA dictionaries produced the same long tails. The `english_us_arpa` model and dictionary end the words at 2.250 s and 2.060 s and label the tail as silence. The comparison changes both the acoustic model and the dictionary because the phone inventories differ.

![Three runs on ISLE clip 01](Results/comparison/ISLE_SESS0131_BLOCKD02_01_sprt1.png)

**A spoken word labelled as a pause.** In `F2BJRLP1` the transcript reads “… one of the toughest. As W B U R’s Margo Melnicove reports …”. The generic dictionary entry `ʈ ə f s ʈ` gives *toughest* one vowel. The aligner ends that word early, places *as* on the rest of it (19.68–19.92 s), and marks 19.92–20.58 s as a pause. The signal level at 20.40–20.55 s is −29 to −31 dBFS, the same as the average of the whole recording, so the spoken *as* was skipped. Both US dictionaries place *as* at 20.35–20.57 s.

![Three runs on F2BJRLP1, 19.2 to 21.5 seconds](Results/comparison/F2BJRLP1_19.2-21.5s.png)

**Letter names compressed.** The generic dictionary has *w* as `d ɐ b` and *u* as a single `ɐ`. The missing syllables of “double-u” are pushed into the next letter, and the vowel of *b* is stretched to 390 ms. With full letter names in the US dictionaries, the two acoustic models agree on all four letters within 20 ms.

Changing only the dictionary corrected the largest mismatch in the long recordings. The ARPAbet pairing addressed the utterance-final silence in the short ISLE clips.

The MFA motor speech tutorial shows how to compare pronunciation variants with the acoustic model held fixed and inspect the resulting phone boundaries in Praat. That design fits the two IPA runs. The original MFA paper compares against human annotations, which would be needed to turn the current agreement statistics into a boundary-accuracy estimate. A few generated entries such as *Dukakis* and *Melnicove* remain pronunciation hypotheses even when dictionary coverage is complete. [MFA tutorial, Case 3](https://github.com/thealk/MFA-Motor-Speech-Tutorial#case-3-toggling-pronunciation-variants-and-weights-to-improve-identification-of-non-native-speech-sequences) · [Original MFA paper](https://people.linguistics.mcgill.ca/~morgan/mcauliffeEtAl2017_mfa.pdf) · [MFA G2P guidance](https://montreal-forced-aligner.readthedocs.io/en/stable/first_steps/index.html#generating-a-pronunciation-dictionary-with-a-pretrained-g2p-model)

## Listening review

| Recording | Listener observations |
| --- | --- |
| ISLE clip 01 | One listener heard “I said white, not bait” and heard *bait* end before 3 s. The other heard “I said wait, not bait” and found the voice unclear. The supplied transcript says *white*. |
| ISLE clip 03 | Both listeners heard “I said bad, not bed.” One found the voice clear and the other heard *bed* end before 3 s. |
| F2BJRLP1 | One listener heard a pause after *toughest*, then a distinct *as* and all four letters of *WBUR*. *B* and *U* sounded a little alike. The listener heard a longer pause after *toughest* than after *reports*. The ARPAbet word intervals show gaps of about 0.38 s and 0.06 s, respectively. The supplied transcript has a period after *toughest* and a comma after *reports*. |

The listening estimates for clips 01 and 03 support the presence of quiet tails. Exact phone endpoints remain unverified.

## Inspection in Praat

Open a WAV from `alignment/corpus/` and the TextGrid with the same name in [Praat](https://www.fon.hum.uva.nl/praat/), select both objects, and choose **View & Edit**. Listen around a narrow window before judging an apparent offset. The observations above combine listening with spectrograms, signal levels, and comparisons between runs. The recordings have no manually annotated reference boundaries.

[MFA installation guide](https://montreal-forced-aligner.readthedocs.io/en/stable/installation.html) · [MFA pretrained models](https://mfa-models.readthedocs.io/)
