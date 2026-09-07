/**
 * Client-side pre-flight checks for video uploads.
 *
 * These mirror the server-side rules in backend/app/api/routes/video_uploads.py
 * (extension allowlist + MAX_UPLOAD_SIZE_MB). The backend remains the real
 * authority - this exists so an obviously-invalid file is rejected before
 * we spend minutes streaming it up only to get a 400/413 back, and so the
 * drop zone enforces the same rule the hint text advertises (the file
 * picker's `accept` attribute doesn't apply to drag-and-drop at all).
 *
 * Keep in sync with the backend defaults if those settings change.
 */

export const ALLOWED_VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
export const MAX_UPLOAD_SIZE_MB = 500

const MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

export function fileExtension(filename) {
  const match = /\.[^.]+$/.exec(String(filename || ''))
  return match ? match[0].toLowerCase() : ''
}

/**
 * Returns a human-readable problem with `file`, or null if it looks
 * acceptable. Deliberately phrased for a non-technical user.
 */
export function validateVideoFile(file) {
  if (!file) return null

  const extension = fileExtension(file.name)
  if (!ALLOWED_VIDEO_EXTENSIONS.includes(extension)) {
    const readable = ALLOWED_VIDEO_EXTENSIONS.join(', ')
    return extension
      ? `"${file.name}" is a ${extension} file. Supported video formats are ${readable}.`
      : `"${file.name}" has no file extension. Supported video formats are ${readable}.`
  }

  if (file.size === 0) {
    return `"${file.name}" is empty (0 bytes). Please choose a different file.`
  }

  if (file.size > MAX_UPLOAD_SIZE_BYTES) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(0)
    return `"${file.name}" is ${sizeMb}MB, which is over the ${MAX_UPLOAD_SIZE_MB}MB limit.`
  }

  return null
}
