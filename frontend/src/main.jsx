import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode> {/*wrapper qui detecte les mauvaises pratiques */}
    <App /> {/*calls the App.jsx that contains the code of the App component . Component = fct jsx */}
  </StrictMode>,
)
