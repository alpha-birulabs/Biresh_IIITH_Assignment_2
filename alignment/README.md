# Part 2a · Montreal Forced Aligner

The pipeline uses the six supplied recordings and their transcripts. It groups them by recording speaker, converts audio to 16 kHz mono PCM WAV, and writes matching UTF-8 `.lab` files. It expands the written initialisms S.J.C. and WBUR and normalizes hyphen-space artifacts such as “seventy- six.” The experiment uses the pretrained global English MFA acoustic model and its matching pronunciation dictionary.

The [report](Alignment_Report.pdf) documents the baseline and supplemented dictionary runs. [TextGrids](TextGrids/) contains the final run with the five added pronunciations. The [interval table](Results/intervals.csv) and [sample visualization](Results/sample_alignment.png) come from those grids.

## Setup and alignment

Install [Miniforge](https://github.com/conda-forge/miniforge) and open a terminal with Conda available. Extract the supplied corpus so `Assignment 2a/wav/` and `Assignment 2a/transcripts/` are available, or pass their parent with `--source`.

```sh
conda create -n aligner -c conda-forge "python=3.12" "montreal-forced-aligner=3.3" "kaldi=*=cpu*" "pytorch=*=cpu*" ffmpeg
conda activate aligner
python alignment/run.py --validate
```

`--prepare-only` builds the audio/transcript pairs without invoking MFA. `--validate` runs MFA's corpus and dictionary coverage checks before alignment. `--dictionary PATH` lets you use a local compatible dictionary. The expected outputs are six TextGrids under `alignment/TextGrids/`, each with word and phone intervals.

For a second run, download the [plain English MFA v3.1.0 dictionary](https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-english_mfa-v3.1.0/english_mfa.dict) and add the manually transcribed pronunciations for rare words and names. These entries use the existing model's phone inventory.

```sh
python alignment/run.py --validate --dictionary .local/english_mfa.dict --lexicon-additions alignment/pronunciations.tsv
```

## Inspection

```sh
python -m pip install -r alignment/requirements.txt
python alignment/inspect_grids.py
```

The script uses the Praat engine to check the complete corpus inventory, tier names, contiguous positive-duration intervals, and durations against the converted WAVs. It exports an interval table, a per-clip overview, and a waveform/spectrogram aligned with the word and phone boundaries. `Results/review_flags.csv` lists phones longer than 0.5 seconds for manual review. This threshold marks inspection candidates and does not establish that a boundary is wrong.

The current outputs flag two final-phone intervals in the ISLE recordings. In clip 01, /t/ runs from 2.050 to 4.125 seconds. In clip 03, /d/ runs from 1.790 to 4.500 seconds. Both long intervals also occur in MFA's internal alignment database. The report discusses the quiet tail assigned to /t/ in clip 01. The [clip 03 visualization](Results/sample_alignment_03.png) shows a release near 1.95 seconds followed by a quiet tail inside /d/. This is further visual evidence of trailing-silence alignment errors. Listening in Praat remains necessary to establish defensible manual corrections.

Regenerate the second visualization with the following command.

```sh
python alignment/inspect_grids.py --sample ISLE_SESS0131_BLOCKD02_03_sprt1 --output .local/boundary-review-03
```

For interactive inspection in [Praat](https://www.fon.hum.uva.nl/praat/), open the corresponding WAV and TextGrid, select both objects, and choose **View & Edit**. Listen around a narrow window before judging an apparent offset. These data have no reference boundary times, so boundary observations are qualitative.

[MFA installation guide](https://montreal-forced-aligner.readthedocs.io/en/stable/installation.html) · [English MFA model and dictionary](https://huggingface.co/MontrealCorpusTools/english_mfa)
