# TESS held-out results

Test set: 420 clips. Accuracy for uniform guessing is 14.29%.

| Modality | Accuracy | Macro F1 | Selected epoch |
| --- | ---: | ---: | ---: |
| Speech temporal | 99.05% | 0.9904 | 4 |
| Text contextual | 14.29% | 0.1081 | 6 |
| Speech + text fusion | 99.52% | 0.9952 | 10 |

## Per-class recall and paired fusion changes

| Emotion | Speech | Text | Fusion | Fusion gains | Fusion losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Angry | 100.00% | 20.00% | 100.00% | 0 | 0 |
| Disgust | 100.00% | 20.00% | 100.00% | 0 | 0 |
| Fear | 100.00% | 43.33% | 100.00% | 0 | 0 |
| Happy | 98.33% | 0.00% | 100.00% | 1 | 0 |
| Neutral | 100.00% | 6.67% | 100.00% | 0 | 0 |
| Pleasant surprise | 95.00% | 10.00% | 96.67% | 1 | 0 |
| Sad | 100.00% | 0.00% | 100.00% | 0 | 0 |

Each gain is a corrected speech-only error. Each loss is a correct speech-only prediction that fusion misclassifies.

## Failure cases

Cases prioritize high-confidence fusion errors, followed by speech-only and text-only errors as needed.

| Clip | True emotion | Speech | Text | Fusion | Fusion confidence |
| --- | --- | --- | --- | --- | ---: |
| YAF_goal_ps | pleasant_surprise | angry | disgust | happy | 80.1% |
| OAF_hall_ps | pleasant_surprise | disgust | fear | disgust | 59.2% |
| OAF_hash_happy | happy | pleasant_surprise | fear | happy | 99.3% |
| OAF_time_ps | pleasant_surprise | disgust | angry | pleasant_surprise | 98.7% |
| OAF_bite_angry | angry | angry | fear | angry | 99.9% |
