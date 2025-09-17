import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { LocalizationProvider } from './context/LocalizationContext.tsx' // 1. Импортируем провайдер

const rootElement = document.getElementById('root')!;
const root = createRoot(rootElement);

root.render(
    <StrictMode>
        {/* 2. Оборачиваем компонент App в наш провайдер */}
        <LocalizationProvider>
            <App />
        </LocalizationProvider>
    </StrictMode>,
)