import { API_URL } from '../api/client'
import { useAuthStore } from '../store/authStore'
import Modal from './Modal'

export default function VideoModal({ recording, onClose }) {
  const token = useAuthStore((s) => s.token)

  if (!recording) return null

  const videoUrl = `${API_URL}/api/recordings/${recording.id}/video?token=${token}`

  return (
    <Modal title={recording.filename || 'Recording'} onClose={onClose} maxWidth="max-w-3xl">
      <video controls autoPlay className="w-full rounded-lg bg-black" src={videoUrl}>
        Your browser does not support the video tag.
      </video>
    </Modal>
  )
}
