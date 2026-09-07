import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

export const useAuthStore = create(
  persist(
    (set, get) => ({
      token: null,
      user: null,

      login: (token, user) => set({ token, user }),

      logout: () => set({ token: null, user: null }),

      isAdmin: () => get().user?.role === 'admin',

      isOperator: () => ['admin', 'operator'].includes(get().user?.role),

      hasRole: (...roles) => roles.includes(get().user?.role),
    }),
    {
      name: 'sentinelcam-auth',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ token: state.token, user: state.user }),
    },
  ),
)
