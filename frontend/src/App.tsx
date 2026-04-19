import { useMemo, useRef, useState } from 'react'

import DestinationInput from './components/DestinationInput'
import ErrorBanner from './components/ErrorBanner'
import ItinerarySummary from './components/ItinerarySummary'
import Settings from './components/Settings'
import StepTimeline from './components/StepTimeline'
import type { Lang } from './i18n'
import { t } from './i18n'
import { fetchCandidates, streamPlan } from './lib/api'
import type {
  AppStatus,
  CandidateWaypoint,
  ErrorInfo,
  PlanningFormState,
  PlanningMode,
  StreamEvent,
  TravelSummary,
} from './types'
import './App.css'

const defaultForm: PlanningFormState = {
  city: '',
  destination: '',
  origin: '',
  tripDurationDays: 1,
  interestsText: '',
  fatigueThresholdMeters: 3000,
  maxSpots: 3,
  transportMode: 'walking',
}

function App() {
  const [mode, setMode] = useState<PlanningMode>('local_explore')
  const [lang, setLang] = useState<Lang>('zh')
  const [status, setStatus] = useState<AppStatus>('idle')
  const [form, setForm] = useState<PlanningFormState>(defaultForm)
  const [candidates, setCandidates] = useState<CandidateWaypoint[]>([])
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<string[]>([])
  const [events, setEvents] = useState<StreamEvent[]>([])
  const [summary, setSummary] = useState<TravelSummary | null>(null)
  const [error, setError] = useState<ErrorInfo | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const interests = useMemo(
    () =>
      form.interestsText
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean),
    [form.interestsText],
  )

  const selectedWaypoints = useMemo(
    () => candidates.filter((candidate) => selectedCandidateIds.includes(candidate.id)),
    [candidates, selectedCandidateIds],
  )

  const resetPlanningState = () => {
    abortRef.current?.abort()
    setEvents([])
    setSummary(null)
    setError(null)
  }

  const handleFieldChange = <K extends keyof PlanningFormState>(key: K, value: PlanningFormState[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  const handleModeChange = (nextMode: PlanningMode) => {
    setMode(nextMode)
    setCandidates([])
    setSelectedCandidateIds([])
    setStatus('idle')
    resetPlanningState()
  }

  const handleGenerateCandidates = async () => {
    resetPlanningState()
    setStatus('loading_candidates')
    try {
      const data = await fetchCandidates({
        mode: 'point_to_point',
        origin: form.origin.trim(),
        destination: form.destination.trim(),
        interests,
        trip_duration_days: form.tripDurationDays,
      })
      setCandidates(data)
      setSelectedCandidateIds(data.slice(0, 2).map((item) => item.id))
      setEvents([
        ...data.map((candidate) => ({
          event: 'candidate_generated',
          payload: {
            name: candidate.name,
            role: candidate.role,
          },
        })),
        {
          event: 'candidate_set_ready',
          payload: { description: `${t(lang, 'candidateReady')} · ${t(lang, 'candidateReadyDesc')}` },
        },
      ])
      setStatus('candidate_ready')
    } catch (err) {
      setError({ type: 'candidate_error', message: err instanceof Error ? err.message : 'Failed to generate candidates' })
      setStatus('error')
    }
  }

  const handleToggleCandidate = (candidateId: string) => {
    setSelectedCandidateIds((prev) =>
      prev.includes(candidateId) ? prev.filter((id) => id !== candidateId) : [...prev, candidateId],
    )
  }

  const handleStartPlanning = async () => {
    resetPlanningState()
    setStatus('running')
    const controller = new AbortController()
    abortRef.current = controller

    try {
      await streamPlan(
        {
          mode,
          destination: form.destination.trim(),
          origin: form.origin.trim(),
          trip_duration_days: form.tripDurationDays,
          interests,
          fatigue_threshold_meters: form.fatigueThresholdMeters,
          max_spots: form.maxSpots,
          selected_waypoints: mode === 'point_to_point' ? selectedWaypoints : [],
          transport_mode: form.transportMode,
          city: form.city.trim(),
        },
        {
          signal: controller.signal,
          onEvent: (event) => {
            if (event.event === 'error') {
              setError(event.payload as unknown as ErrorInfo)
              setStatus('error')
              return
            }
            if (event.event === 'planning_completed') {
              setSummary(event.payload as unknown as TravelSummary)
              setStatus('completed')
            }
            setEvents((prev) => [...prev, event])
          },
        },
      )
    } catch (err) {
      if (controller.signal.aborted) return
      setError({ type: 'network_error', message: err instanceof Error ? err.message : 'Connection failed' })
      setStatus('error')
    }
  }

  return (
    <div className="app-shell">
      <div className="background-orb orb-left" />
      <div className="background-orb orb-right" />

      <header className="hero">
        <div className="hero-topbar">
          <div className="segmented-control">
            <button
              className={mode === 'local_explore' ? 'active' : ''}
              onClick={() => handleModeChange('local_explore')}
              type="button"
            >
              {t(lang, 'localMode')}
            </button>
            <button
              className={mode === 'point_to_point' ? 'active' : ''}
              onClick={() => handleModeChange('point_to_point')}
              type="button"
            >
              {t(lang, 'routeMode')}
            </button>
          </div>

          <div className="hero-actions">
            <div className="lang-switch">
              <button className={lang === 'zh' ? 'active' : ''} onClick={() => setLang('zh')} type="button">
                中
              </button>
              <button className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')} type="button">
                EN
              </button>
            </div>
            <button className="ghost-button" type="button" onClick={() => setSettingsOpen(true)}>
              {t(lang, 'settings')}
            </button>
          </div>
        </div>

        <div className="hero-copy">
          <div>
            <p className="eyebrow">Fatigue-Aware Agent</p>
            <h1>{t(lang, 'title')}</h1>
            <p className="hero-subtitle">{t(lang, 'subtitle')}</p>
          </div>
          <div className="hero-mode-copy">
            <h2>{mode === 'local_explore' ? t(lang, 'localMode') : t(lang, 'routeMode')}</h2>
            <p>{mode === 'local_explore' ? t(lang, 'localDescription') : t(lang, 'routeDescription')}</p>
          </div>
        </div>
      </header>

      <main className="workspace">
        <div className="workspace-left">
          <DestinationInput
            mode={mode}
            form={form}
            status={status}
            candidates={candidates}
            selectedCandidateIds={selectedCandidateIds}
            lang={lang}
            onFieldChange={handleFieldChange}
            onGenerateCandidates={handleGenerateCandidates}
            onToggleCandidate={handleToggleCandidate}
            onStartPlanning={handleStartPlanning}
          />

          {error && (
            <ErrorBanner
              error={error}
              lang={lang}
              onRetry={mode === 'point_to_point' && !candidates.length ? handleGenerateCandidates : handleStartPlanning}
              onOpenSettings={() => setSettingsOpen(true)}
            />
          )}
        </div>

        <div className="workspace-right">
          <StepTimeline events={events} lang={lang} />
          <ItinerarySummary summary={summary} lang={lang} />

          <section className="dashboard-grid">
            <article className="panel-card placeholder-card">
              <p className="eyebrow">{t(lang, 'dashboard')}</p>
              <h2>{t(lang, 'mapPlaceholder')}</h2>
              <p>{t(lang, 'mapPlaceholderDesc')}</p>
            </article>
            <article className="panel-card placeholder-card">
              <p className="eyebrow">{t(lang, 'dashboard')}</p>
              <h2>{t(lang, 'fatiguePanel')}</h2>
              <p>{t(lang, 'fatiguePanelDesc')}</p>
              <div className="fatigue-stats">
                <span>{form.fatigueThresholdMeters} m</span>
                <span>{summary?.total_distance_meters ?? 0} m</span>
              </div>
            </article>
          </section>
        </div>
      </main>

      <Settings open={settingsOpen} onClose={() => setSettingsOpen(false)} lang={lang} />
    </div>
  )
}

export default App
