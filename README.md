# IIIT Hyderabad LTRC Assignment 2

- [Part 2a · Forced alignment with Montreal Forced Aligner](alignment/README.md) · [PDF report](alignment/Alignment_Report.pdf)
- [Part 2b · Speech, text, and multimodal emotion recognition](emotion/README.md) · [report](emotion/REPORT.md)

## Part 2a in brief

The six supplied recordings are aligned three times: with the `english_mfa` acoustic model and its generic dictionary, with the same model and the US English dictionary, and with the `english_us_arpa` model and dictionary. Words missing from a dictionary are covered by hand-written entries in the first run and by pretrained G2P models in the other two. One command runs the alignments, the Praat checks, the comparison and the PDF report.

The final alignment is the [six ARPAbet TextGrids](alignment/TextGrids_us_arpa/). The two other runs provide comparison grids and interval tables. The ARPAbet pair separates trailing silence from the final phones and places the spoken *as* after the audible pause.

## Part 2b in brief

Three classifiers predict seven emotions on TESS from speech, from text, and from both. They share one word-disjoint split with 420 held-out clips.

The submission retains the log-Mel BiGRU, character BiGRU and joint representation fusion. The [matched five-seed comparison](emotion/REPORT.md#matched-comparison-with-repo-inspired-models) includes MFCC-style speech, TF-IDF text, and score fusion baselines. The accuracy differences between the speech and fusion alternatives were at most 0.05 percentage points on average.

[Learned modality weighting](emotion/REPORT.md#learned-weights-versus-concatenation) averaged 99.33%, compared with 99.38% for the submitted concatenation model on the same split and seeds.

| Model | Test accuracy | Macro F1 |
| --- | ---: | ---: |
| Speech only | 99.05% | 0.990 |
| Text only | 14.29% | 0.108 |
| Speech + text fusion | 99.52% | 0.995 |

Every TESS transcript is spoken in all seven emotions, so text alone cannot beat uniform guessing (14.29%). Over five training seeds speech averages 99.43% and fusion 99.38%, so fusion does not add accuracy on this dataset. The report covers the architecture choices, the comparison, per-emotion results, failure cases, learned-representation plots, and a robustness study over training seeds, text ablations and noisy audio.

## Run

Run all commands from this directory. Each part's README lists its setup and commands. The supplied alignment recordings and the TESS archive are not redistributed here.

Run the regression checks after installing the emotion dependencies.

```sh
python -m unittest discover -s tests -v
```
