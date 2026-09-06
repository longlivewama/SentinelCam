# SentinelCam Backend

FastAPI backend for the SentinelCam AI surveillance platform: JWT-authenticated
REST API, MJPEG live camera streaming, event-triggered recording, and
YOLOv8-based fall / violence / crowd / abandoned-object detection.

## Setup

1. **Create and activate a virtual environment**

   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create a PostgreSQL database**

   ```bash
   createdb sentinelcam
   # or, from psql:
   # CREATE DATABASE sentinelcam;
   # CREATE USER sentinelcam WITH PASSWORD 'sentinelcam';
   # GRANT ALL PRIVILEGES ON DATABASE sentinelcam TO sentinelcam;
   ```

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and fill in at least `DATABASE_URL` and `JWT_SECRET_KEY`.
   SMTP settings are only required if you want real alert emails to be sent
   (`send_alert_email` fails safe and just logs a warning otherwise).

5. **Create the first admin user**

   ```bash
   python seed_admin.py
   ```

   This reads `ADMIN_EMAIL` / `ADMIN_PASSWORD` from the environment if set,
   otherwise it prompts interactively. Tables are created automatically
   (no separate migration step - this project uses `Base.metadata.create_all`
   rather than Alembic).

6. **Run the API**

   ```bash
   uvicorn app.main:app --reload
   ```

   The API is served under `http://localhost:8000/api/...`; interactive docs
   are available at `http://localhost:8000/docs`. CORS is pre-configured for
   the Vite frontend dev server (`http://localhost:5173` and
   `http://localhost:3000`).

## Notes

- **Recordings/snapshots** are written to `storage/recordings/` and
  `storage/snapshots/` under `backend/` by default (configurable via
  `RECORDINGS_DIR` / `SNAPSHOTS_DIR`).
- **Video codec**: recordings are written with the `avc1` (H.264) fourcc
  first, falling back to `mp4v` if your OpenCV/ffmpeg build can't open an
  H.264 encoder (common on stock `opencv-python` wheels, which often don't
  bundle a licensed H.264 encoder). Check the server logs for which codec
  was used for a given recording; `mp4v` clips are less broadly compatible
  with `<video>` playback (notably Safari) than `avc1`.
- **YOLO model weights** (`yolov8n-pose.pt`, `yolov8n.pt`) are downloaded
  automatically by `ultralytics` on first use if not already present
  locally, and are gitignored (`*.pt`).
- **Violence detection** is an MVP heuristic (pose-keypoint velocity +
  directional variance), not a trained model - see the docstring in
  `app/services/detection/violence_detection.py` for details and the
  `VIOLENCE_MODEL_PATH` config hook for swapping in a trained model later.
- **Crowd detection** counts YOLO "person" detections rather than using a
  density-map model (e.g. CSRNet) - see
  `app/services/detection/crowd_detection.py`.
- **Abandoned object tracking** uses a small hand-rolled centroid tracker
  rather than DeepSORT/ByteTrack - see
  `app/services/detection/abandoned_object_detection.py`.
