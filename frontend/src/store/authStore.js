import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

export const useAuthStore = create(
  persist(
    (set, get) => ({
      token: null,
      user: null,

      login: (token, user) => set({ token, user }),

      logout: () => set({ token: null, user: null }),

      isAdmin: () => Boolean(get().user?.is_admin),
    }),
    {
      name: 'sentinelcam-auth',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ token: state.token, user: state.user }),
    },
  ),
)
