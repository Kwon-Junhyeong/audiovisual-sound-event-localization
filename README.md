# Audiovisual Sound Event Localization

DCASE 2026 Challenge, Task 3, audiovisual track: from four microphones and a 360-degree camera, find which of 13 sound classes are active and where each one is on the panorama, 10 times per second. Third place in the team ranking, and the highest AP75 of all 15 systems in the official evaluation. Research intern work at the KIST AI and Robotics Institute, Humanoid Research Center (2026). Technical report: [report/Kwon_KIST_task3_technical_report.pdf](report/Kwon_KIST_task3_technical_report.pdf).

![System](docs/system.png)

## Diagnosis first

With no confidence threshold, the official baseline's detector already finds 71% of the events (IoU 0.25), yet it scores mAP 0.0003. The ranking fails, not the features. So the detector keeps up to 100 candidates per frame, down to score 0.01 (`ranking/dump_detections.py`).

## Rerank by match probability

A gradient-boosting model reads features of each candidate (detector score, energy-map statistics, box geometry, co-detections, SRP-PHAT direction) and predicts whether it will match a reference event. That probability replaces the detector's score (`ranking/rerank.py`, `ranking/srp_features.py`).

## Output the whole energy region

The reference field is dense, so each box writes its predicted energy over its extent as a set of points instead of a single peak (`ranking/submit.py`). The soft IoU with the reference rises from 0.27 to 0.84.

## Results

On the development set, mask output and reranking raised the system mAP from 0.0013 to 0.021 without retraining the detector, 70 times the official baseline. In the official evaluation, the system had the highest AP75 of all 15 systems (0.0009), and the team placed third.

## Run

The code builds on the official baseline repository. Set `BASELINE_DIR` to it and `DCASE_LABELS` to the development labels.

```sh
python detector/train.py            # detector on the dev clips
python ranking/dump_detections.py CHECKPOINT N_CLIPS
python ranking/srp_features.py TAG
python ranking/rerank.py TAG
python ranking/submit.py TAG
```
