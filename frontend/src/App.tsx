import { useState, useRef, useCallback } from 'react'
import DestinationInput from './components/DestinationInput'
import StepTimeline from './components/StepTimeline'
import ItinerarySummary from './components/ItinerarySummary'
import Settings from './components/Settings'
import ErrorBanner from './components/ErrorBanner'
import type { StepEvent, AppStatus, ErrorInfo } from './types'
import type { Lang } from './i18n'
import { t } from './i18n'
import './App.css'

function App() {
  const [steps, setSteps] = useState<StepEvent[]>([])
  const [status, setStatus] = useState<AppStatus>('idle')
  const [errorInfo, setErrorInfo] = useState<ErrorInfo | null>(null)
  const [lang, setLang] = useState<Lang>('zh')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)
  const lastRequestRef = useRef<{ destination: string; origin: string } | null>(null)

  const startPlanning = useCallback((destination: string, origin: string) => {
    setSteps([])
    setStatus('running')
    setErrorInfo(null)
    lastRequestRef.current = { destination, origin }

    const params = new URLSearchParams({ destination })
    if (origin) params.set('origin', origin)

    const es = new EventSource(`/api/plan/stream?${params.toString()}`)
    eventSourceRef.current = es

    es.addEventListener('node_update', (e) => {
      const data: StepEvent = JSON.parse(e.data)
      setSteps((prev) => [...prev, data])
    })

    es.addEventListener('done', () => {
      setStatus('done')
      es.close()
    })

    es.addEventListener('error', (e) => {
      if (e instanceof MessageEvent) {
        try {
          const data: ErrorInfo = JSON.parse(e.data)
          setErrorInfo(data)
        } catch {
          setErrorInfo({ type: 'unknown_error', message: e.data })
        }
      } else {
        setErrorInfo({ type: 'network_error', message: lang === 'zh' ? '连接中断' : 'Connection lost' })
      }
      setStatus('error')
      es.close()
    })

    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) return
      setErrorInfo({ type: 'network_error', message: lang === 'zh' ? '连接中断' : 'Connection lost' })
      setStatus('error')
      es.close()
    }
  }, [lang])

  const handleRetry = useCallback(() => {
    if (lastRequestRef.current) {
      startPlanning(lastRequestRef.current.destination, lastRequestRef.current.origin)
    }
  }, [startPlanning])

  return (
    <div className="app">
      <header>
        <div className="header-bar">
          <div className="lang-switch">
            <button
              className={lang === 'zh' ? 'active' : ''}
              onClick={() => setLang('zh')}
            >中</button>
            <button
              className={lang === 'en' ? 'active' : ''}
              onClick={() => setLang('en')}
            >EN</button>
          </div>
          <button className="settings-btn" onClick={() => setSettingsOpen(true)}>
            ⚙️ {t(lang, 'settings')}
          </button>
        </div>
        <h1>{t(lang, 'title')}</h1>
        <p className="subtitle">{t(lang, 'subtitle')}</p>
      </header>

      <main>
        <DestinationInput
          onStart={startPlanning}
          disabled={status === 'running'}
          lang={lang}
        />

        {status === 'error' && errorInfo && (
          <ErrorBanner
            error={errorInfo}
            lang={lang}
            onRetry={handleRetry}
            onOpenSettings={() => setSettingsOpen(true)}
          />
        )}

        <StepTimeline steps={steps} lang={lang} />

        {status === 'done' && <ItinerarySummary steps={steps} lang={lang} />}
      </main>

      <Settings
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        lang={lang}
      />
    </div>
  )
}

export default App
