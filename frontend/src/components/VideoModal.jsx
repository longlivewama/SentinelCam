import MediaVideo from './MediaVideo'
import Modal from './Modal'

export default function VideoModal({ recording, onClose }) {
  if (!recording) return null

  return (
    <Modal title={recording.filename || 'Recording'} onClose={onClose} maxWidth="max-w-3xl">
      {/* The clip URL carries a short-lived, clip-scoped media token
          rather than the session JWT - see lib/mediaToken.js. */}
      <MediaVideo kind="recording" id={recording.id} autoPlay />
    </Modal>
  )
}
