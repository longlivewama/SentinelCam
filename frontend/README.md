# SentinelCam Frontend

Dark, professional surveillance-monitoring console built with React (Vite), React Router,
Zustand, Axios, and Tailwind CSS.

## Getting Started

```bash
npm install
cp .env.example .env
npm run dev
```

The app runs at `http://localhost:5173` by default. Set `VITE_API_URL` in `.env` to point at the
SentinelCam backend (defaults to `http://localhost:8000`); the realtime WebSocket URL is derived
from this automatically (`http(s)` → `ws(s)`, same host/port, `/api/ws/events`).

## Testing

```bash
npm run lint
npm run test        # vitest + @testing-library/react, jsdom environment
npm run test:watch  # watch mode
npm run build
```

Tests mock the `apiClient`/`realtime` modules rather than hitting a real backend — see
`src/pages/*.test.jsx` for auth-flow tests (login/signup/forgot/reset, route guards by role) and
`src/pages/VideoUpload.test.jsx` for the upload-workflow tests (drag-drop, progress, delete,
empty/loading/error states).

## Production Build

```bash
npm run build
npm run preview   # optional: preview the production build locally
```

## Project Structure

- `src/api/client.js` — shared Axios instance with auth header + 401 handling.
- `src/lib/realtime.js` — WebSocket connection (auto-reconnect with backoff) + pub/sub for
  realtime alert/camera-status/upload-progress events.
- `src/store/authStore.js` — logged-in user/token/role, persisted to `localStorage`; the frontend
  re-fetches `/api/auth/me` on every app load rather than trusting the cached role indefinitely
  (important after an admin changes someone's role, or after a backend schema change).
- `src/store/toastStore.js` / `src/components/ToastContainer.jsx` — global success/error/warning
  notifications.
- `src/store/alertsBadgeStore.js` — unread-alerts count shown on the Alerts nav item.
- `src/components/` — route guards (`ProtectedRoute`, `AdminRoute`, `OperatorRoute`), navbar,
  charts (`BarChart`), stat tiles, camera tile/form, recording row, video/modal components.
- `src/pages/` — Login, Signup, ForgotPassword, ResetPassword, Dashboard, Cameras, CameraDetail,
  Recordings, Alerts, Analytics, VideoUpload, SystemStatus, AdminUsers (Users), Settings.

## Notes

- Live camera streams, recording video/download links, the video-upload source-video player, and
  the WebSocket connection all carry the JWT as a `?token=` query parameter rather than an
  `Authorization` header — browsers cannot attach custom headers to `<img src>`, `<video src>`,
  `<a href>`, or the native WebSocket API. All other API calls go through the shared Axios
  instance with a `Authorization: Bearer <token>` header.
- Role-based UI (which nav items, which action buttons show) mirrors the backend's RBAC but is a
  UX convenience only — the backend independently enforces every permission, so a hidden button
  here is not the actual security boundary.
