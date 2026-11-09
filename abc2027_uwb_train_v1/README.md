# ABC2027 UWB Challenge

Detect and temporally localize falls in a continuous, unsegmented ultra-wideband radar stream. A stream mixes activities of daily living with falls. The task is to predict the start and the end of each fall. Scoring is event-level: correct detections, missed falls, false alarms, start and end errors, temporal IoU (tIoU), and detection latency.

## Installation

Python 3.10 or later is required. From the root of the extracted package:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The `pip install -e .` step makes the `abc2027` package importable from anywhere (for example `python -m abc2027.evaluate`). Without it, run commands from the package root with `PYTHONPATH=src`.

## Recording setup

The recordings come from the LIARA apartment at UQAC, about 40 m2. Three Novelda XeThru X4M200 radars are used, serial numbers 29171,29444,30722. Each CIR frame has 368 range bins. Frames are aligned to a common grid at 50 Hz. Localization lines in the source files are not part of the streams.

## Source recordings and published streams

Source fall files are labeled from the file name (`Chute01` to `Chute04`, plus other and empty rooms). Activity files cover 15 daily activities. There is no shared person identifier between the fall sessions and the activity corpus: subject splits are disjoint inside each corpus only.

The published package contains continuous streams built by concatenation:

- 13 training streams (6.18 h)
- 5 validation streams (2.05 h)

`streams_events.csv` lists 478 annotated fall events for those streams. `streams_composition.csv` lists the source segments of the training and validation streams only. The official test streams are not in this package. The public generator seed plans training and validation. It does not reconstruct the official test streams.

## Package layout

```
streams/train/stream_00.h5
streams/val/stream_00.h5
streams_events.csv
streams_composition.csv
sample_submission.csv
configs/streams.yaml
configs/baseline.yaml
src/abc2027/evaluate.py
src/abc2027/baseline.py
scripts/build_streams.py
```

Further files named stream_XX.h5 sit beside stream_00.h5. A segments directory holds the aligned source arrays used by the generator.

Each HDF5 file stores `cir` with shape `(3, T, 368)` as float32, `t` in seconds, and `radar_serials`.

```python
from pathlib import Path
import h5py
root = Path("streams/val/stream_00.h5")
if not root.exists():
    root = Path("release/abc2027_uwb_train_v1/streams/val/stream_00.h5")
with h5py.File(root, "r") as handle:
    print(handle["cir"].shape)
```

## What counts as a fall

A fall event starts at the onset of the movement and ends when the person is stable on the floor. Time spent lying on the floor after that is not part of the fall. Bounds were placed from the change in the static CIR profile, on the radar where that change is largest, after a short high-pass of the background. On the published streams the annotated interval is unchanged by the 0.2 s crossfade used at segment joins.

Start and end times carry an uncertainty of ±0.5 to 1 s, because the radar clocks are integer Unix seconds. The evaluator reports the fraction of matched events inside ±0.5 s and ±1 s. Those tolerances do not change the tIoU match.

A frame whose strongest radar residual is below the residual gate 5.12e-06 is low quality. This mainly affects fall dataset 3. Low-quality falls and falls marked `needs_review` stay in the streams.

## Split

Training, validation, and the held-out test use different subjects. The activity subjects and the fall subjects are not the same people, so the split does not pair one person across the two corpora. Test recordings, test labels, and the test segment list are withheld.

## Submission, metrics, and ranking

A submission is a CSV with columns `stream_id`, `t_start`, `t_end`, and `score`. An optional `t_alarm` is the time of the first alarm. `t_start` must be strictly before `t_end`, and both times must lie inside the stream.

**Stream identifiers.** Each stream file `streams/<split>/stream_XX.h5` has the identifier `<split>_XX`, where `<split>` is the folder name and `XX` the two-digit index: `train_00` to `train_12`, `val_00` to `val_04`, and, when the test set is released, `test_00` to `test_04`. The `stream_id` column of a submission must use these identifiers exactly.

Predictions are matched one-to-one, greedy by descending score. A prediction is a true positive when its tIoU with an evaluable fall is at least the threshold. The primary theta is 0.3. The report also reports 0.1, 0.3, 0.5, 0.7.

The ranking score is the mean event F1 at theta 0.1, 0.3 and 0.5. Ties are broken by a lower median absolute start error at theta 0.3, then by fewer false alarms per hour at theta 0.3. Mean tIoU, start and end errors, F1 at each theta including 0.7, and latency are reported and do not break ties. Latency on matches at theta 0.3 is `t_alarm` minus the ground-truth start when `t_alarm` is present, otherwise the predicted start minus the ground-truth start.

Events with `ignore = true` are neither true positives nor false negatives. Matching against evaluable events happens first. Only then is an unmatched prediction dropped, neither as a true positive nor as a false alarm, when it overlaps an ignore zone widened by ±2 s. A prediction that already matches an evaluable fall stays a true positive even if it also touches a nearby ignore zone.

`--ignore-flagged` treats `quality = low` and `needs_review` as ignore zones. Validation can then be scored under the same rule as the official test. Without the flag, only the `ignore` column is used. On the published validation file that column is false for every event, so the flag is required for a test-like score.

Evaluate from this package with:

`python -m abc2027.evaluate --gt streams_events.csv --pred sample_submission.csv --streams streams --ignore-flagged --out report.json`

## Baselines

Scores below were produced by `src/abc2027/baseline.py` and measured with the official evaluator on validation. The thresholds were tuned on validation. They are not an estimate of test performance.

The energy baseline keeps the residual after a 3 s exponential background removal, thresholds it at the percentile 98 of that stream, merges runs separated by less than 0.8 s, and drops runs shorter than 0.6 s. With `--ignore-flagged`, the energy baseline F1 mean 0.315 (0.358 / 0.338 / 0.249 / 0.060) at theta 0.1 / 0.3 / 0.5 / 0.7.

The learned baseline uses 200-frame windows (4 s) with a hop of 50 frames, motion features after removing the window mean, and a balanced logistic regression. The operating point is a score threshold 0.85. With `--ignore-flagged`, the logistic baseline F1 mean 0.258 (0.593 / 0.167 / 0.014 / 0.000). The drop toward theta 0.5 is expected for a 4 s window placed on a fall of about 1.6 s: temporal IoU then stays near 0.4.

`sample_submission.csv` is the energy baseline on the validation streams.

## Known limits

Falls were simulated on a mat. The three mat positions are fixed. Activities were recorded elsewhere in the apartment. A pre-fall window, ending 0.5 s before the annotated start, is still easy to tell from activities: pre-fall margin versus ADL is 1.000, and versus other or empty recordings is 0.839. That context is present in the raw streams. Two corrections were measured and not applied. EMA time constants of 2 s and 8 s leave AUC 0.923 and 0.960 against ADL, because the filter can be inverted up to a constant. Subtracting the per-segment median CIR carries the floor-time profile back into the margin: segment-median removal leaves 0.964 against ADL and 0.935 against other recordings.

Dataset 3 includes many recordings below the residual gate above, so those falls are low confidence.

Bounds are uncertain by ±0.5 to 1 s. Subject identity is not aligned between the fall corpus and the activity corpus. Streams are synthetic concatenations: segments are joined with a 0.2 s crossfade, activity files are cut at random points, and fall files are cropped with random margins around the annotated event so that a junction is not itself a fall marker.

## Rules

The use of the source UWB recordings from LIARA (the original fall and activity corpora, or any external copy thereof) is strictly prohibited for both training and inference. Top-ranking teams will be required to provide their code and models for verification.

## Download and submission

The dataset is available on the ABC2027 challenge website. 
Release dates for the test set and submission deadlines: To be announced on the challenge website.

## License and citation

This dataset is distributed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). You can view a copy of this license at [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/).

Cite:

- Maitre, Bouchard and Gaboury, IEEE JBHI 2021, https://doi.org/10.1109/JBHI.2020.3027967
- Imbeault-Nepton et al., GoodIT 2023, https://doi.org/10.1145/3582515.3609535
- Abudalfa and Bouchard, Transportation Research Procedia 2025, https://doi.org/10.1016/j.trpro.2025.03.039
- Abudalfa and Bouchard, Applied Intelligence 2025, https://doi.org/10.1007/s10489-024-06156-9

Contacts: Habba's Ngodjou Doukaga (hndoukaga@etu.uqac.ca) and Kevin Bouchard (Kevin_Bouchard@uqac.ca).
