# Part 2b · TESS emotion recognition

Three classifiers predict the same seven emotions from speech, text, and speech plus text. The experiment uses [the assigned TESS Kaggle archive](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).

- [Experiment report](REPORT.md)
- [Accuracy and per-class results](Results/accuracy.md)

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
```

Run all commands from the repository root. Training uses CUDA when available and otherwise runs on CPU. Each train/test entry point supports `--help`. Features are cached under `emotion/data/` and checkpoints under `emotion/checkpoints/`. Evaluation writes metrics and per-clip predictions to `emotion/Results/`. The summarizer writes accuracy tables, paired changes, failure cases, confusion matrices, and representation plots there.

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

| Block | Implementation | Rationale |
| --- | --- | --- |
| Speech preprocessing | Mono 16 kHz audio, DC removal, peak scaling, RMS-based edge trimming | Standardizes sampling and removes long quiet edges while retaining internal pauses and relative dynamics. |
| Speech features | 64 log-Mel bands from 25 ms frames and a 10 ms hop, standardized with training statistics | Models the spectral envelope and its evolution on a compact frequency scale. |
| Speech temporal modelling | Bidirectional GRU with masked attention, using every fourth frame after feature extraction | Models changes in voice quality and timing over the utterance, with attention over valid frames only. |
| Text preprocessing and features | Lowercase reconstructed transcript, character tokenization, learned 24-dimensional embeddings | Supports unseen target words without a fixed word vocabulary. |
| Text contextual modelling | Bidirectional GRU and masked attention | Pools character context into an utterance representation. |
| Fusion | Concatenate pooled speech and text vectors, project to 64 dimensions, normalize and apply GELU | Lets an end-to-end classifier learn the relative contribution of both modalities. |
| Classifier | Dropout and a seven-logit linear head, trained with cross-entropy and AdamW | Uses a common prediction objective across modalities. Validation macro F1 selects the checkpoint. |

The paired comparison counts gains and losses on identical held-out clips. Each PCA panel is fitted independently to visualize one representation space. Overlapping text points are expected because every transcript has the same representation across its seven labels.
