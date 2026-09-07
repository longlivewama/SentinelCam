import { create } from 'zustand'

let nextId = 1

export const useToastStore = create((set) => ({
  toasts: [],

  push: (toast) => {
    const id = nextId++
    const entry = { id, type: 'info', duration: 5000, ...toast }
    set((state) => ({ toasts: [...state.toasts, entry] }))
    if (entry.duration > 0) {
      setTimeout(() => {
        set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }))
      }, entry.duration)
    }
    return id
  },

  dismiss: (id) => set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
}))

export const toast = {
  success: (message) => useToastStore.getState().push({ type: 'success', message }),
  error: (message) => useToastStore.getState().push({ type: 'error', message }),
  info: (message) => useToastStore.getState().push({ type: 'info', message }),
  warning: (message) => useToastStore.getState().push({ type: 'warning', message }),
}
