# Word-boundary agreement between MFA runs

Every run aligns the same six recordings. A shift is the absolute difference between two runs for the same word token. The corpus has no reference boundaries, so a shift shows disagreement, not error.

## Per recording

| Recording | Words | Phones (mfa_generic) | Pauses (mfa_generic) | Longest phone s (mfa_generic) | Phones (mfa_us) | Pauses (mfa_us) | Longest phone s (mfa_us) | Phones (arpa_us) | Pauses (arpa_us) | Longest phone s (arpa_us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F2BJRLP1 | 73 | 295 | 12 | 0.39 | 303 | 11 | 0.23 | 308 | 14 | 0.23 |
| F2BJRLP2 | 74 | 339 | 11 | 0.31 | 338 | 8 | 0.31 | 345 | 10 | 0.24 |
| F2BJRLP3 | 82 | 359 | 15 | 0.29 | 366 | 13 | 0.29 | 372 | 13 | 0.29 |
| ISLE_SESS0131_BLOCKD02_01_sprt1 | 5 | 13 | 1 | 2.08 | 13 | 1 | 2.08 | 13 | 3 | 0.21 |
| ISLE_SESS0131_BLOCKD02_02_sprt1 | 5 | 11 | 2 | 0.35 | 11 | 2 | 0.35 | 12 | 3 | 0.35 |
| ISLE_SESS0131_BLOCKD02_03_sprt1 | 5 | 13 | 1 | 2.71 | 13 | 1 | 2.71 | 13 | 2 | 0.26 |

Pauses are unlabeled word intervals. A longest phone above 0.5 s marks a boundary to review.

## Pairwise agreement

### mfa_generic vs mfa_us

| Speaker | Word tokens | Boundary | Median (ms) | Mean (ms) | Within 20 ms | Within 50 ms | Largest (ms) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| F2BJ | 229 | start | 0 | 11 | 90% | 95% | 670 |
| F2BJ | 229 | end | 0 | 12 | 90% | 95% | 650 |
| SESS0131 | 15 | start | 0 | 3 | 93% | 100% | 50 |
| SESS0131 | 15 | end | 0 | 3 | 93% | 100% | 50 |
| All | 244 | start | 0 | 11 | 90% | 95% | 670 |
| All | 244 | end | 0 | 12 | 91% | 95% | 650 |

Largest word-end shifts (mfa_us minus mfa_generic):

| Recording | Word | mfa_generic end (s) | mfa_us end (s) | Shift (ms) |
| --- | --- | ---: | ---: | ---: |
| F2BJRLP1 | as | 19.920 | 20.570 | +650 |
| F2BJRLP1 | toughest | 19.680 | 20.000 | +320 |
| F2BJRLP1 | w | 20.700 | 20.910 | +210 |
| F2BJRLP1 | b | 21.120 | 21.010 | -110 |
| F2BJRLP3 | of | 16.190 | 16.300 | +110 |
| F2BJRLP3 | not | 28.040 | 27.940 | -100 |

### mfa_generic vs arpa_us

| Speaker | Word tokens | Boundary | Median (ms) | Mean (ms) | Within 20 ms | Within 50 ms | Largest (ms) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| F2BJ | 229 | start | 10 | 17 | 87% | 94% | 680 |
| F2BJ | 229 | end | 10 | 16 | 88% | 95% | 640 |
| SESS0131 | 15 | start | 10 | 31 | 73% | 80% | 110 |
| SESS0131 | 15 | end | 20 | 324 | 53% | 60% | 2440 |
| All | 244 | start | 10 | 18 | 86% | 93% | 680 |
| All | 244 | end | 10 | 35 | 86% | 93% | 2440 |

Largest word-end shifts (arpa_us minus mfa_generic):

| Recording | Word | mfa_generic end (s) | arpa_us end (s) | Shift (ms) |
| --- | --- | ---: | ---: | ---: |
| ISLE_SESS0131_BLOCKD02_03_sprt1 | bed | 4.500 | 2.060 | -2440 |
| ISLE_SESS0131_BLOCKD02_01_sprt1 | bait | 4.125 | 2.250 | -1875 |
| F2BJRLP1 | as | 19.920 | 20.560 | +640 |
| F2BJRLP1 | toughest | 19.680 | 19.980 | +300 |
| F2BJRLP1 | w | 20.700 | 20.900 | +200 |
| F2BJRLP2 | clerk | 28.290 | 28.470 | +180 |

### mfa_us vs arpa_us

| Speaker | Word tokens | Boundary | Median (ms) | Mean (ms) | Within 20 ms | Within 50 ms | Largest (ms) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| F2BJ | 229 | start | 10 | 9 | 94% | 98% | 250 |
| F2BJ | 229 | end | 0 | 8 | 94% | 98% | 150 |
| SESS0131 | 15 | start | 10 | 34 | 73% | 80% | 160 |
| SESS0131 | 15 | end | 20 | 328 | 53% | 60% | 2440 |
| All | 244 | start | 10 | 11 | 93% | 97% | 250 |
| All | 244 | end | 10 | 28 | 92% | 95% | 2440 |

Largest word-end shifts (arpa_us minus mfa_us):

| Recording | Word | mfa_us end (s) | arpa_us end (s) | Shift (ms) |
| --- | --- | ---: | ---: | ---: |
| ISLE_SESS0131_BLOCKD02_03_sprt1 | bed | 4.500 | 2.060 | -2440 |
| ISLE_SESS0131_BLOCKD02_01_sprt1 | bait | 4.125 | 2.250 | -1875 |
| ISLE_SESS0131_BLOCKD02_01_sprt1 | white | 1.490 | 1.330 | -160 |
| F2BJRLP2 | clerk | 28.320 | 28.470 | +150 |
| ISLE_SESS0131_BLOCKD02_02_sprt1 | new | 1.290 | 1.420 | +130 |
| ISLE_SESS0131_BLOCKD02_01_sprt1 | not | 1.690 | 1.800 | +110 |
