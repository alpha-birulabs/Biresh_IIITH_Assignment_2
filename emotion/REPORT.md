# Part 2b · Multimodal emotion recognition on TESS

## Architecture and evaluation

The speech model standardizes 64-band log-Mel features using training-set statistics. A bidirectional GRU and attention pooling summarize the acoustic sequence. The text model learns character embeddings and uses a second bidirectional GRU with attention to encode each transcript. The fusion model concatenates those two representations and projects them before classification. The three models use seven-way cross-entropy, and validation macro F1 selects the checkpoint. The [implementation notes](README.md#experimental-design) describe the choices for each block.

TESS contains two speakers reading each of 200 target words in seven emotions, always in the same carrier phrase. The 140/30/30-word split keeps target words disjoint and has 60 recordings per emotion on the test set. Both speakers appear in all splits. [Original TESS dataset description](https://doi.org/10.5683/SP2/E8H2MF)

## Held-out results

| Model | Correct / total | Accuracy |
| --- | ---: | ---: |
| Speech only | 416 / 420 | 99.05% |
| Text only | 60 / 420 | 14.29% |
| Speech + text | 418 / 420 | 99.52% |

The speech model recovered every angry, disgust, fear, neutral, and sad clip. Pleasant surprise had the lowest speech recall at 95%, with two errors into disgust and one into angry. The happy error went into pleasant surprise. This mix suggests that models need cues beyond vocal activation to distinguish some of the performances.

Fusion recovered one happy clip and one pleasant-surprise clip, with zero new errors. That is a gain of two clips, or 0.48 percentage points on this split. Each transcript occurs with each emotion and both speakers, so any deterministic transcript-only classifier gets 1/7 accuracy. The small fusion gain could arise from how the text representation conditions the acoustic decision or from differences in model capacity and optimization. A single split with two speakers cannot distinguish these explanations.

## Clip-level errors

| Clip | Reference | Speech prediction | Fusion prediction | Observation |
| --- | --- | --- | --- | --- |
| `OAF_hash_happy` | Happy | Pleasant surprise | Happy | Fusion recovered the only happy clip missed by speech. Its predicted probability for happy was 99.3%. |
| `OAF_hall_ps` | Pleasant surprise | Disgust | Disgust | Both models made the same confusion. The predicted probability for disgust fell from 89.1% to 59.2%. |
| `OAF_time_ps` | Pleasant surprise | Disgust | Pleasant surprise | Fusion corrected the same kind of speech error seen on `hall`. The speaker and emotion match, so the correction is specific to the clip. |
| `YAF_goal_ps` | Pleasant surprise | Angry | Happy | Fusion changed the predicted class and still missed pleasant surprise. The true label was hardest in both acoustic models. |

## Learned representations

The [PCA panels](Results/plots/representations.png) show the pooled speech and text encoder outputs and the projected fusion features. Speech places many happy and pleasant-surprise clips near each other. Fusion shows a tighter pleasant-surprise cluster with a few outliers. Angry and happy overlap in the first two fusion components despite perfect recall in both categories, since the classifier uses the full representation. The text panel collapses the 14 recordings of each transcript to one position because the model receives the same characters for all of them.

The [confusion matrices](Results/plots/confusion.png) show the held-out prediction counts. The [results tables](Results/accuracy.md) include exact recall, macro F1, and paired gains and losses.
