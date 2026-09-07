import { useEffect, useState } from 'react'
import apiClient from '../api/client'
import { useAuthStore } from '../store/authStore'
import CameraTile from '../components/CameraTile'
import CameraForm from '../components/CameraForm'
import Modal from '../components/Modal'

export default function Cameras() {
  const isOperator = useAuthStore((s) => s.isOperator())
  const [cameras, setCameras] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showAddModal, setShowAddModal] = useState(false)

  const fetchCameras = async () => {
    setLoading(true)
    setError('')
    try {
      const { data } = await apiClient.get('/api/cameras')
      setCameras(data)
    } catch {
      setError('Failed to load cameras.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchCameras()
  }, [])

  const handleCreate = async (payload) => {
    await apiClient.post('/api/cameras', payload)
    setShowAddModal(false)
    fetchCameras()
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Cameras</h1>
          <p className="mt-1 text-sm text-slate-400">
            {cameras.length} camera{cameras.length === 1 ? '' : 's'} configured
          </p>
        </div>
        {isOperator && (
          <button onClick={() => setShowAddModal(true)} className="sc-btn-primary">
            + Add Camera
          </button>
        )}
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-status-error/30 bg-status-error/10 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="py-24 text-center text-slate-500">Loading cameras…</div>
      ) : cameras.length === 0 ? (
        <div className="sc-card py-24 text-center text-slate-500">
          No cameras configured yet.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {cameras.map((camera) => (
            <CameraTile key={camera.id} camera={camera} />
          ))}
        </div>
      )}

      {showAddModal && (
        <Modal title="Add Camera" onClose={() => setShowAddModal(false)}>
          <CameraForm
            submitLabel="Create Camera"
            onSubmit={handleCreate}
            onCancel={() => setShowAddModal(false)}
          />
        </Modal>
      )}
    </div>
  )
}
