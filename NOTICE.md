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

Review each of these before any commercial deployment or redistribution.
