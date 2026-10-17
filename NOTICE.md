# Third-Party Notices

SentinelCam itself is released under the [MIT License](LICENSE). It depends on and integrates
with third-party software and data that carry their own licence terms, which are **not** granted
by that MIT licence:

- **Ultralytics YOLOv8** (`ultralytics`) is distributed under **AGPL-3.0**. Using it in a
  networked service can impose source-availability obligations. Review the Ultralytics licensing
  terms — or obtain their commercial licence — before any commercial deployment.
- **Model weights auto-downloaded at runtime** (e.g. `yolov8n.pt`, `yolov8n-pose.pt`) are governed
  by their publisher's terms.
- **Training datasets referenced by `ml/`** are not redistributed in this repository and remain
  subject to the licences of their original sources — see [`ml/README.md`](ml/README.md).
- **Pose model.** `yolov8n-pose.pt` is Ultralytics' pretrained COCO-Keypoints model (AGPL-3.0, as
  above). The keypoint format and skeleton it uses are COCO's; COCO annotations are CC BY 4.0.
- **Application screenshots in `docs/images/`** that show video frames (`video-analysis-*`,
  `multi-person-tracking`, `person-tracking-timeline`, `fall-event-details`,
  `camera-live-monitoring`, `mobile-video-analysis`) contain frames from the **UR Fall Detection
  Dataset** — B. Kwolek, M. Kepski, *Human fall detection on embedded platform using depth maps and
  wireless accelerometer*, Computer Methods and Programs in Biomedicine 117(3), 2014 — which is
  licensed **CC BY-NC-SA 4.0**. Those images are therefore made available under CC BY-NC-SA 4.0,
  not the MIT licence, and may not be used commercially. `multi-person-tracking.png` shows two URFD
  clips placed side by side in one frame. The overlays on them are SentinelCam's real output.

Review each of these before any commercial deployment or redistribution.
