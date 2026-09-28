import { createRoot } from 'react-dom/client'
import App from './App'
import './styles/base.css'
import './styles/asterra.css'

try {
  const saved = localStorage.getItem('dw-theme')
  const theme = saved === 'light' ? 'light' : 'dark'
  document.documentElement.dataset.theme = theme
  // Keep browser chrome in sync on first paint — previously the meta tag kept
  // its static value until the first in-app toggle.
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute('content', theme === 'light' ? '#eae5d6' : '#05090c')
} catch {
  document.documentElement.dataset.theme = 'dark'
}

createRoot(document.getElementById('root')).render(<App />)
