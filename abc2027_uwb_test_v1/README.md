# ABC2027 UWB Challenge - Test set (v1)

This package contains the unlabeled test streams of the ABC2027 UWB Fall Detection Challenge.
It complements the training package abc2027_uwb_train_v1 (task description, evaluation script,
baselines and starter notebook). Please read the README of the training package first.

## Content
- streams/test/stream_00.h5 ... stream_04.h5: 5 continuous, non-segmented streams, about 2.07 hours in total.
- Same format as the training and validation streams: cir (3 x T x 368, float32), t (seconds, 50 Hz), radar_serials (29171, 29444, 30722).
- No labels and no metadata about the composition of the streams.

## Stream identifiers
streams/test/stream_XX.h5 has the identifier test_XX (test_00 to test_04). Use these identifiers in the stream_id column of your submission.

## Submission
One CSV file with the columns stream_id, t_start, t_end, score (optional t_alarm), one line per detected fall,
with times in seconds as in the t dataset of each stream. Check your file with the validator of the official evaluation script.
Ranking: mean event-level F1 at temporal IoU thresholds 0.1, 0.3 and 0.5 (see the training package README for ignored zones and tie-breakers).
Deadline and procedure: to be announced on the challenge website.

## Rules
Using the original LIARA UWB source recordings (fall and activity corpora, or any external copy) for training or inference is not allowed.
Top-ranked teams will be asked to provide their code and models for verification.

## License
CC BY-NC 4.0 - see LICENSE.

## Contact
Habba's Ngodjou Doukaga (hndoukaga@etu.uqac.ca), Kevin Bouchard (Kevin_Bouchard@uqac.ca), LIARA Laboratory, Universite du Quebec a Chicoutimi.
