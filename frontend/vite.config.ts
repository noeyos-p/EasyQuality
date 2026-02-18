import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/rag': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
      '/agent': 'http://localhost:8000',
      '/graph': 'http://localhost:8000',
      '/llm': 'http://localhost:8000',
      '/evaluate': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/models': 'http://localhost:8000',
      '/test': 'http://localhost:8000',
      '/onlyoffice': 'http://localhost:8000',
      '/auth': 'http://localhost:8000',
      '/processing': 'http://localhost:8000',
    },
  },
})
