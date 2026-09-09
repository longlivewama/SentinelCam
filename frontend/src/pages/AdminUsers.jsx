import { useCallback, useEffect, useState } from 'react'
import apiClient from '../api/client'
import Modal from '../components/Modal'
import Pagination from '../components/Pagination'
import { useAuthStore } from '../store/authStore'
import { toast } from '../store/toastStore'
import { formatDateTime } from '../lib/format'

const ROLE_BADGE = {
  admin: 'border-accent-blue/30 bg-accent-blue/10 text-accent-blue',
  operator: 'border-status-ok/30 bg-status-ok/10 text-status-ok',
  viewer: 'border-surface-600 bg-surface-800 text-slate-400',
}

function AddUserForm({ onSubmit, onCancel }) {
  const [form, setForm] = useState({ email: '', password: '', full_name: '', role: 'viewer' })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const update = (field, value) => setForm((f) => ({ ...f, [field]: value }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setSubmitting(true)
    try {
      await onSubmit(form)
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to create user.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div>
        <label htmlFor="new-user-full-name" className="sc-label">Full Name</label>
        <input
          id="new-user-full-name"
          required
          className="sc-input"
          value={form.full_name}
          onChange={(e) => update('full_name', e.target.value)}
        />
      </div>
      <div>
        <label htmlFor="new-user-email" className="sc-label">Email</label>
        <input
          id="new-user-email"
          required
          type="email"
          className="sc-input"
          value={form.email}
          onChange={(e) => update('email', e.target.value)}
        />
      </div>
      <div>
        <label htmlFor="new-user-password" className="sc-label">Password</label>
        <input
          id="new-user-password"
          required
          type="password"
          minLength={8}
          className="sc-input"
          value={form.password}
          onChange={(e) => update('password', e.target.value)}
        />
      </div>
      <div>
        <label htmlFor="new-user-role" className="sc-label">Role</label>
        <select id="new-user-role" className="sc-input" value={form.role} onChange={(e) => update('role', e.target.value)}>
          <option value="viewer">Viewer - read-only access</option>
          <option value="operator">Operator - manage cameras &amp; alerts</option>
          <option value="admin">Admin - full access incl. user management</option>
        </select>
      </div>

      {error && (
        <div className="rounded-lg border border-status-error/30 bg-status-error/10 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="mt-2 flex justify-end gap-3">
        <button type="button" onClick={onCancel} className="sc-btn-secondary">
          Cancel
        </button>
        <button type="submit" disabled={submitting} className="sc-btn-primary">
          {submitting ? 'Creating…' : 'Create User'}
        </button>
      </div>
    </form>
  )
}

const PAGE_SIZE = 20

export default function AdminUsers() {
  const currentUser = useAuthStore((s) => s.user)
  const [users, setUsers] = useState([])
  const [pageInfo, setPageInfo] = useState({ page: 1, pages: 0, total: 0 })
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showAddModal, setShowAddModal] = useState(false)
  const [busyId, setBusyId] = useState(null)

  const fetchUsers = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const { data } = await apiClient.get('/api/admin/users', {
        params: { page, page_size: PAGE_SIZE },
      })
      setUsers(data.items)
      setPageInfo({ page: data.page, pages: data.pages, total: data.total })
    } catch {
      setError('Failed to load users.')
    } finally {
      setLoading(false)
    }
  }, [page])

  useEffect(() => {
    fetchUsers()
  }, [fetchUsers])

  const handleCreate = async (payload) => {
    await apiClient.post('/api/admin/users', payload)
    setShowAddModal(false)
    toast.success('User created.')
    fetchUsers()
  }

  const handleToggle = async (user) => {
    setBusyId(user.id)
    try {
      const { data } = await apiClient.put(`/api/admin/users/${user.id}/toggle`)
      setUsers((prev) => prev.map((u) => (u.id === user.id ? data : u)))
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Failed to update user.')
    } finally {
      setBusyId(null)
    }
  }

  const handleRoleChange = async (user, role) => {
    setBusyId(user.id)
    try {
      const { data } = await apiClient.put(`/api/admin/users/${user.id}/role`, { role })
      setUsers((prev) => prev.map((u) => (u.id === user.id ? data : u)))
      toast.success(`${user.email} is now ${role}.`)
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Failed to change role.')
    } finally {
      setBusyId(null)
    }
  }

  const handleDelete = async (user) => {
    if (!window.confirm(`Delete user "${user.email}"? This cannot be undone.`)) return
    setBusyId(user.id)
    try {
      await apiClient.delete(`/api/admin/users/${user.id}`)
      setUsers((prev) => prev.filter((u) => u.id !== user.id))
      toast.success('User deleted.')
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Failed to delete user.')
    } finally {
      setBusyId(null)
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Users</h1>
          <p className="mt-1 text-sm text-slate-400">Manage platform accounts, roles, and access.</p>
        </div>
        <button onClick={() => setShowAddModal(true)} className="sc-btn-primary">
          + Add User
        </button>
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-status-error/30 bg-status-error/10 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="py-24 text-center text-slate-500">Loading users…</div>
      ) : users.length === 0 ? (
        <div className="sc-card py-24 text-center text-slate-500">No users found.</div>
      ) : (
        <div className="sc-card overflow-x-auto">
          <table className="w-full min-w-[720px] text-left">
            <thead>
              <tr className="border-b border-surface-700 text-xs uppercase tracking-wide text-slate-500">
                <th className="px-4 py-3 font-medium">Name</th>
                <th className="px-4 py-3 font-medium">Email</th>
                <th className="px-4 py-3 font-medium">Role</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Created</th>
                <th className="px-4 py-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user) => {
                const isSelf = user.id === currentUser?.id
                return (
                  <tr key={user.id} className="border-b border-surface-800 last:border-b-0 hover:bg-surface-800/40">
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-200">{user.full_name}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-400">{user.email}</td>
                    <td className="whitespace-nowrap px-4 py-3">
                      {isSelf ? (
                        <span className={`sc-badge border capitalize ${ROLE_BADGE[user.role]}`}>{user.role}</span>
                      ) : (
                        <select
                          aria-label={`Change role for ${user.email}`}
                          className="sc-input py-1 text-xs capitalize"
                          value={user.role}
                          disabled={busyId === user.id}
                          onChange={(e) => handleRoleChange(user, e.target.value)}
                        >
                          <option value="viewer">Viewer</option>
                          <option value="operator">Operator</option>
                          <option value="admin">Admin</option>
                        </select>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3">
                      <button
                        onClick={() => handleToggle(user)}
                        disabled={busyId === user.id || isSelf}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition disabled:opacity-50 ${
                          user.is_active ? 'bg-status-ok/70' : 'bg-surface-600'
                        }`}
                        aria-label="Toggle active status"
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
                            user.is_active ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                      <span className="ml-2 text-xs text-slate-400">
                        {user.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-400">
                      {formatDateTime(user.created_at)}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3">
                      <button
                        onClick={() => handleDelete(user)}
                        disabled={busyId === user.id || isSelf}
                        className="sc-btn-danger px-3 py-1.5 text-xs"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      <Pagination
        page={pageInfo.page}
        pages={pageInfo.pages}
        total={pageInfo.total}
        pageSize={PAGE_SIZE}
        onChange={setPage}
        busy={loading}
        noun="users"
      />

      {showAddModal && (
        <Modal title="Add User" onClose={() => setShowAddModal(false)}>
          <AddUserForm onSubmit={handleCreate} onCancel={() => setShowAddModal(false)} />
        </Modal>
      )}
    </div>
  )
}
