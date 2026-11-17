# Part 2b · TESS emotion recognition

Three classifiers predict the same seven emotions from speech, text, and speech plus text. The experiment uses [the assigned TESS Kaggle archive](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).

- [Experiment report](REPORT.md): architecture decisions, comparison, analysis, failure cases and representation plots
- [Accuracy and per-class results](Results/accuracy.md)
- [Robustness study](Results/robustness.md): training seeds, text ablations and noisy audio
- [Plots](Results/plots/): confusion matrices, learned representations, accuracy under noise

## Reproduce

Install the dependencies in a Python 3.12 environment. Download and extract the archive into a directory such as `.local/tess`.

```sh
python -m pip install -r emotion/requirements.txt
python -m emotion.data_prep .local/tess
python -m emotion.models.speech_pipeline.train
python -m emotion.models.speech_pipeline.test
python -m emotion.models.text_pipeline.train
python -m emotion.models.text_pipeline.test
python -m emotion.models.fusion_pipeline.train
python -m emotion.models.fusion_pipeline.test
python -m emotion.summarize
python -m emotion.robustness --source .local/tess
```

Run all commands from the repository root. Training uses CUDA when available and otherwise runs on CPU. Each train/test entry point supports `--help`. Features are cached under `emotion/data/` and checkpoints under `emotion/checkpoints/`. Evaluation writes metrics and per-clip predictions to `emotion/Results/`. The summarizer writes accuracy tables, paired changes, failure cases, silhouette scores, confusion matrices, and PCA and t-SNE plots of the learned representations there. The robustness study trains four more seeds of every model on the same split, re-evaluates the fusion models with the text replaced, and tests speech and fusion on audio with added white noise. Its extra checkpoints go to `.local/seed-study/`.

## Predict from new inputs

The three small trained checkpoints are included under `emotion/checkpoints/`. Use the same audio preprocessing and text normalization through the prediction command.

```sh
python -m emotion.predict --mode speech --audio path/to/clip.wav
python -m emotion.predict --mode text --text "say the word boat"
python -m emotion.predict --mode fusion --audio path/to/clip.wav --text "say the word boat"
```

Each command returns JSON with the selected emotion and all seven probabilities. For fusion, supply the actual transcript of the audio. The models were evaluated on TESS's controlled recordings from two speakers, so the reported accuracy describes that setting. The text model's 14.29% accuracy follows from the balanced repeated phrases in TESS.

New training runs save training settings, hashes of all prepared splits, and validation history alongside the best checkpoint. Evaluation rejects data that differs from those recorded hashes and checks that target words remain disjoint. The original included checkpoints predate this metadata and remain supported. They reproduce the results above with the documented seed 42 split.

The local verification used Python 3.12, CPU PyTorch 2.11.0, NumPy 2.5.5, SciPy 1.17.1, SoundFile 0.14.0, scikit-learn 1.8.0, and Matplotlib 3.10.8. Re-evaluating all three supplied checkpoints reproduced every saved metric and prediction row. Additional checks covered raw-audio versus cached-feature predictions, batch padding invariance, a one-epoch training/evaluation round trip, invalid audio/text rejection, changed-data detection, overlapping split rejection, and checkpoint preservation after a simulated write failure.

## Experimental design

The archive contains two copies of the dataset. Preprocessing checks identical content before deduplicating, and maps the single `OA_bite_neutral` filename to speaker OAF. The resulting 2,800 clips form a complete 200-word × 2-speaker × 7-emotion design. We reconstruct the transcript as “say the word {target word}” and use only that string on the text branch.

We split by target word with seed 42, holding each word and all its recordings in one split. Training, validation, and test contain 1,960, 420, and 420 clips. Both speakers occur in each split. The experiment measures generalization to held-out words from these two speakers. Every held-out transcript occurs in all seven emotions in equal numbers, so a deterministic transcript-only classifier has 1/7 accuracy by construction. The dataset does not test whether text carries emotion in less constrained conversation. [Original TESS dataset description](https://doi.org/10.5683/SP2/E8H2MF)

The [report](REPORT.md#a-architecture-decisions) gives the architecture of every block and the reason for it.

The paired comparison counts gains and losses on identical held-out clips. Each PCA and t-SNE panel is fitted independently to visualize one representation space. The text panels show one point per transcript, because every transcript has the same representation across its seven labels.
