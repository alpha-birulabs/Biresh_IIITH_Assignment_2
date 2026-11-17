# Robustness of the modality comparison

All models use the same word-disjoint split and the same 420 held-out clips. Only the seed for weight initialisation and batch order changes. Seed 42 is the released checkpoint set.

## Training seeds

| Seed | Speech | Text | Fusion | Fusion fixes speech errors | Fusion adds errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 99.05% | 14.29% | 99.52% | 2 | 0 |
| 1 | 99.52% | 14.29% | 99.05% | 0 | 2 |
| 2 | 99.29% | 14.29% | 99.52% | 2 | 1 |
| 3 | 99.52% | 14.29% | 99.29% | 0 | 1 |
| 4 | 99.76% | 14.29% | 99.52% | 0 | 1 |
| Mean ± SD | 99.43% ± 0.27% | 14.29% ± 0.00% | 99.38% ± 0.21% | 0.8 | 1.0 |

Fusion is more accurate than speech in 2 of 5 seeds, equal in 0 and less accurate in 3.

Errors by emotion, summed over the 5 seeds (300 clips per emotion):

| Model | Angry | Disgust | Fear | Happy | Neutral | Pleasant surprise | Sad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Speech | 0 | 1 | 0 | 1 | 0 | 10 | 0 |
| Fusion | 0 | 2 | 0 | 0 | 0 | 11 | 0 |

## What the fusion model does with the text

The trained fusion models are evaluated again with the text input replaced. The audio is unchanged.

| Text given to the fusion model | Accuracy (mean ± SD) | Predictions that change (mean of 420) |
| --- | ---: | ---: |
| matched transcript | 99.38% ± 0.21% | 0.0 |
| carrier phrase only | 99.43% ± 0.27% | 0.2 |
| transcript of another clip | 99.43% ± 0.27% | 0.2 |

## White noise added to the test audio

Noise is added to the 16 kHz signal before trimming and feature extraction. The models were trained on clean audio only.

| Signal-to-noise ratio | Speech (mean ± SD) | Fusion (mean ± SD) |
| --- | ---: | ---: |
| clean | 99.43% ± 0.27% | 99.38% ± 0.21% |
| 30 dB | 86.95% ± 2.90% | 91.86% ± 4.26% |
| 20 dB | 53.43% ± 8.68% | 56.33% ± 4.93% |
| 10 dB | 29.95% ± 3.31% | 41.67% ± 8.24% |
| 0 dB | 22.38% ± 6.86% | 19.00% ± 5.51% |
