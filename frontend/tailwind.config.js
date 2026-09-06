/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          950: '#05070a',
          900: '#0b0f14',
          850: '#0f141b',
          800: '#141a23',
          700: '#1c242f',
          600: '#2a3441',
          500: '#3d4a5a',
        },
        accent: {
          cyan: '#22d3ee',
          blue: '#3b82f6',
        },
        status: {
          ok: '#22c55e',
          warn: '#f59e0b',
          error: '#ef4444',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(34, 211, 238, 0.15), 0 0 20px rgba(34, 211, 238, 0.08)',
      },
    },
  },
  plugins: [],
}
