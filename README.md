# IIIT Hyderabad LTRC Assignment 2

- [Part 2a · Forced alignment with Montreal Forced Aligner](alignment/README.md)
- [Part 2b · Speech, text, and multimodal emotion recognition](emotion/README.md)

Run commands from this directory. Use the supplied archive for the alignment recordings and the linked Kaggle archive for TESS.

The local implementation includes six MFA TextGrids, alignment inspection outputs and a PDF report, three trained emotion classifiers with train/test and prediction commands, and experiment results with plots and error analysis.

Test accuracy on the shared 420-clip split is 99.05% for speech, 14.29% for text, and 99.52% for fusion. Target words are disjoint across splits, with both speakers present in each split.

Two final-phone boundaries are flagged for manual listening in the alignment README.

Run the regression checks after installing the emotion dependencies.

```sh
python -m unittest discover -s tests -v
```
