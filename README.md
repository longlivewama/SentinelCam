# SentinelCam

A full-stack AI surveillance platform: live multi-camera monitoring in the browser, with automatic AI detection of falls, violence, abandoned objects, and crowding — each triggering a recorded clip and an email alert.

- `backend/` — FastAPI + PostgreSQL + OpenCV + YOLOv8 (see `backend/README.md`)
- `frontend/` — React + Vite + Tailwind + Zustand (see `frontend/README.md`)

## Quick start

```bash
# 1. Backend
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
createdb sentinelcam                     # requires a running local PostgreSQL
cp .env.example .env                     # fill in DATABASE_URL, JWT_SECRET_KEY, SMTP_* etc.
python seed_admin.py                     # creates the first admin user
uvicorn app.main:app --reload            # http://localhost:8000  (docs at /docs)

# 2. Frontend (separate terminal)
cd frontend
npm install
cp .env.example .env                     # VITE_API_URL=http://localhost:8000
npm run dev                              # http://localhost:5173
```

Log in with the admin account you created, then add a camera (an RTSP URL for an IP camera, or a device index like `0` for a USB webcam) from the Cameras page.

## Architecture notes

**Auth.** JWT (24h expiry), bcrypt-hashed passwords. Every protected backend route accepts the token either as an `Authorization: Bearer` header (used by all JSON API calls from the frontend's Axios client) or as a `?token=` query parameter. The query-param path exists because browsers cannot attach custom headers to `<img src>`, `<video src>`, or plain `<a href>` downloads — so the MJPEG stream, video playback, and file-download endpoints are linked to directly with the token in the URL, while everything else goes through the header.

**Streaming.** Each camera gets one lazily-started background thread that owns the `cv2.VideoCapture` connection, decodes frames, and holds the latest JPEG in a shared buffer. The `/stream` endpoint serves that shared buffer as `multipart/x-mixed-replace` to as many simultaneous viewers as connect, without opening a second capture connection per viewer.

**Recording.** The same capture thread keeps a rolling buffer of the last ~3 seconds of raw frames. When a detector fires an event, the recording engine snapshots that buffer, appends ~3 more seconds of live frames, and writes the combined clip to disk (`avc1`/H.264 first, falling back to `mp4v` if the local OpenCV build lacks that codec), then records it in the database and sends an email alert with a snapshot attached.

## Where this deviates from a "textbook" implementation, and why

Three of the four detectors described in the original spec assume production-grade models trained on specific research datasets (RWF-2000 for violence, ShanghaiTech via CSRNet for crowd density, DeepSORT for tracking). Training those from scratch needs the actual datasets plus GPU time that isn't available in this environment, so each was replaced with a practical, honestly-scoped equivalent that fits the same interface and can be swapped for a trained model later without touching the rest of the system:

| Detector | Spec called for | What's actually implemented | Upgrade path |
|---|---|---|---|
| Fall | YOLOv8-Pose, 3-signal (aspect ratio / keypoint alignment / hip velocity) | **Implemented as specified**, real pretrained YOLOv8-Pose (COCO weights, auto-downloaded) | Fine-tune on UR Fall / Le2i / FALL-UP if accuracy needs improve beyond COCO-pretrained |
| Violence | CNN+LSTM trained on RWF-2000 | Heuristic reusing the same pose keypoints: erratic high-velocity wrist/elbow motion between people in close proximity | Swap in a trained temporal model via the `VIOLENCE_MODEL_PATH` config hook already wired into `services/detection/violence_detection.py` |
| Crowd counting | CSRNet density-map estimation | YOLOv8 person-class detection + count, smoothed over frames, thresholded per camera | Swap in CSRNet for very dense/occluded scenes where individual-person detection breaks down |
| Abandoned object | YOLO + DeepSORT | YOLO object detection + a small hand-rolled centroid tracker (nearest-centroid matching, first-seen/last-seen timers, proximity-to-person reset) | Swap in DeepSORT/ByteTrack for more robust ID persistence through occlusion |

Everything else — camera CRUD, JWT auth with the two roles, MJPEG streaming, the recording pipeline, the full DB schema (including the recommended `Events` table), all 16 API endpoints, and all 6 frontend pages with the dark cyan/blue theme — was built to the full spec.

## What was actually verified (not just written)

- Backend: booted end-to-end against a real database, every endpoint exercised by hand (auth incl. header vs. query-token, camera CRUD, admin user CRUD + disable/toggle, ranged video streaming, downloads), and the full detect → record → persist → email pipeline was run live against a real webcam with real YOLO inference — a genuine `abandoned_object` event fired and produced a valid, playable MP4 plus snapshot.
- Frontend: `npm run build` production build is clean, and every API call in the codebase was cross-checked against the backend contract by hand (path, method, header-auth vs. query-token pattern).

Not yet verified: an actual RTSP IP camera (only a USB webcam was available to test with), real email delivery against a live SMTP server (code path is exercised but fails safe/silently if SMTP isn't configured), and the full logged-in browser flow click-by-click end-to-end (backend and frontend were verified independently, not yet driven together in a browser).
