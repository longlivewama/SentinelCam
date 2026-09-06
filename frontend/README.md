# SentinelCam Frontend

Dark, professional surveillance-monitoring console built with React (Vite), React Router,
Zustand, Axios, and Tailwind CSS.

## Getting Started

```bash
npm install
cp .env.example .env
npm run dev
```

The app runs at `http://localhost:5173` by default. Set `VITE_API_URL` in `.env` to point
at the SentinelCam backend (defaults to `http://localhost:8000`).

## Production Build

```bash
npm run build
npm run preview   # optional: preview the production build locally
```

## Project Structure

- `src/api/client.js` — shared Axios instance with auth header + 401 handling.
- `src/store/authStore.js` — Zustand store for the logged-in user/token, persisted to `localStorage`.
- `src/components/` — shared UI: route guards, navbar, camera tile, recording row, modals, forms.
- `src/pages/` — top-level routed pages (Login, Cameras, Camera Detail, Recordings, Admin Users, Settings).

## Notes

- Live camera streams and recording video/download links are plain `<img>`/`<video>`/`<a>`
  elements pointed at the backend with a `?token=` query parameter (browsers cannot attach
  custom headers to these requests), matching the backend's supported auth pattern for those
  endpoints. All other API calls go through the shared Axios instance with a
  `Authorization: Bearer <token>` header.
