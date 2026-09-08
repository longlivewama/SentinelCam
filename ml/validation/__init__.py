"""
Video-level operational validation for the SentinelCam fall detector.

The model's published metrics (precision 0.846 / recall 0.800 / mAP@50
0.877) are per-FRAME on still images. The product's claim is per-INCIDENT
on video: does a fall that happened raise an alert, how quickly, and how
often does ordinary activity raise one that shouldn't. Those are different
quantities, and nothing in the repository measured the second set.

This package measures them. It drives the real production inference path
(app.services.detection.video_scan, the same loop the upload analyser
runs) over a labelled corpus of short clips, matches predicted incidents
against annotated ones under an explicit documented rule, and reports
incident recall, incident precision, false alerts per hour and detection
latency - plus a sweep over the two thresholds that govern the trade-off.

See README.md in this directory for the dataset layout and how to run it.
"""
