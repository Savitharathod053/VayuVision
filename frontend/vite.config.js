import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/stations': 'http://127.0.0.1:8000',
      '/current': 'http://127.0.0.1:8000',
      '/forecast': 'http://127.0.0.1:8000',
      '/explain': 'http://127.0.0.1:8000',
      '/dispersion': 'http://127.0.0.1:8000',
      '/history': 'http://127.0.0.1:8000',
      '/correlation': 'http://127.0.0.1:8000',
      '/health-advisory': 'http://127.0.0.1:8000',
      '/nearby-sources': 'http://127.0.0.1:8000',
      '/alerts': 'http://127.0.0.1:8000',
      '/movement-forecast': 'http://127.0.0.1:8000',
      '/decision-support': 'http://127.0.0.1:8000',
      '/model-trust': 'http://127.0.0.1:8000',
      '/aerosol': 'http://127.0.0.1:8000',
      '/fires': 'http://127.0.0.1:8000',
      '/data-status': 'http://127.0.0.1:8000',
      '/simulate': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000'
    }
  }
})
