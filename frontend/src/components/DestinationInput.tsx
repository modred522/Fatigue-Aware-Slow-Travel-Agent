import { useState } from 'react'

import type { Lang } from '../i18n'
import { t } from '../i18n'
import { validateLocation } from '../lib/api'
import type {
  AppStatus,
  CandidateWaypoint,
  PlanningFormState,
  PlanningMode,
  TransportMode,
} from '../types'

interface Props {
  mode: PlanningMode
  form: PlanningFormState
  status: AppStatus
  candidates: CandidateWaypoint[]
  selectedCandidateIds: string[]
  lang: Lang
  onFieldChange: <K extends keyof PlanningFormState>(key: K, value: PlanningFormState[K]) => void
  onGenerateCandidates: () => void
  onToggleCandidate: (candidateId: string) => void
  onStartPlanning: () => void
}

export default function DestinationInput({
  mode,
  form,
  status,
  candidates,
  selectedCandidateIds,
  lang,
  onFieldChange,
  onGenerateCandidates,
  onToggleCandidate,
  onStartPlanning,
}: Props) {
  const isRouteMode = mode === 'point_to_point'
  const candidatesReady = candidates.length > 0
  const isBusy = status === 'loading_candidates' || status === 'running'

  // Location validation state
  const [validationStatus, setValidationStatus] = useState<'idle' | 'validating' | 'valid' | 'invalid'>('idle')
  const [validationMessage, setValidationMessage] = useState('')

  const handleValidateLocation = async () => {
    if (!form.city.trim() || !form.destination.trim()) {
      setValidationStatus('invalid')
      setValidationMessage('Please enter both city and location')
      return
    }

    setValidationStatus('validating')
    setValidationMessage('')

    try {
      const result = await validateLocation({
        city: form.city.trim(),
        location: form.destination.trim(),
      })

      if (result.valid) {
        setValidationStatus('valid')
        setValidationMessage(result.formatted_address || 'Location found')
      } else {
        setValidationStatus('invalid')
        setValidationMessage(result.message || 'Location not found')
      }
    } catch (err) {
      setValidationStatus('invalid')
      setValidationMessage(err instanceof Error ? err.message : 'Validation failed')
    }
  }

  const getValidationClass = () => {
    switch (validationStatus) {
      case 'valid':
        return 'validation-success'
      case 'invalid':
        return 'validation-error'
      case 'validating':
        return 'validation-pending'
      default:
        return ''
    }
  }

  const canStartPlanning = isRouteMode
    ? !isBusy && form.city.trim() && form.destination.trim() && form.origin.trim() && candidatesReady
    : !isBusy && form.city.trim() && form.destination.trim()

  return (
    <section className="panel-card control-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">{t(lang, 'planningPanel')}</p>
          <h2>{isRouteMode ? t(lang, 'routeMode') : t(lang, 'localMode')}</h2>
        </div>
        <span className="mode-pill">{t(lang, 'modeLabel')}</span>
      </div>

      <div className="field-grid">
        {/* City input */}
        <label className="field">
          <span>{t(lang, 'cityLabel')}</span>
          <small>{t(lang, 'cityHelp')}</small>
          <input
            value={form.city}
            onChange={(e) => {
              onFieldChange('city', e.target.value)
              setValidationStatus('idle')
              setValidationMessage('')
            }}
            placeholder={t(lang, 'cityPlaceholder')}
            disabled={isBusy}
          />
        </label>

        {/* Destination/Origin inputs based on mode */}
        {isRouteMode ? (
          <>
            <label className="field">
              <span>{t(lang, 'originLabel')}</span>
              <input
                value={form.origin}
                onChange={(e) => onFieldChange('origin', e.target.value)}
                placeholder={t(lang, 'originPlaceholder')}
                disabled={isBusy}
              />
            </label>
            <label className="field">
              <span>{t(lang, 'destinationLabel')}</span>
              <input
                value={form.destination}
                onChange={(e) => onFieldChange('destination', e.target.value)}
                placeholder={t(lang, 'destinationPlaceholder')}
                disabled={isBusy}
              />
            </label>
          </>
        ) : (
          <label className="field">
            <span>{t(lang, 'destinationLabel')}</span>
            <div className="input-with-button">
              <input
                value={form.destination}
                onChange={(e) => {
                  onFieldChange('destination', e.target.value)
                  setValidationStatus('idle')
                  setValidationMessage('')
                }}
                placeholder={t(lang, 'destinationPlaceholder')}
                disabled={isBusy}
              />
              <button
                type="button"
                className="ghost-button small"
                onClick={handleValidateLocation}
                disabled={isBusy || !form.city.trim() || !form.destination.trim()}
              >
                {validationStatus === 'validating'
                  ? t(lang, 'validating')
                  : t(lang, 'validateLocation')}
              </button>
            </div>
            {validationStatus !== 'idle' && (
              <span className={`validation-message ${getValidationClass()}`}>
                {validationStatus === 'valid' && '✓ '}
                {validationStatus === 'invalid' && '✗ '}
                {validationMessage}
              </span>
            )}
          </label>
        )}

        <label className="field field-full">
          <span>{t(lang, 'interestsLabel')}</span>
          <input
            value={form.interestsText}
            onChange={(e) => onFieldChange('interestsText', e.target.value)}
            placeholder={t(lang, 'interestsPlaceholder')}
            disabled={isBusy}
          />
        </label>

        <label className="field">
          <span>{t(lang, 'durationLabel')}</span>
          <input
            type="number"
            min={1}
            max={14}
            value={form.tripDurationDays}
            onChange={(e) => onFieldChange('tripDurationDays', Number(e.target.value))}
            disabled={isBusy}
          />
        </label>

        <label className="field">
          <span>{t(lang, 'thresholdLabel')}</span>
          <input
            type="number"
            min={500}
            max={20000}
            step={100}
            value={form.fatigueThresholdMeters}
            onChange={(e) => onFieldChange('fatigueThresholdMeters', Number(e.target.value))}
            disabled={isBusy}
          />
        </label>

        <label className="field">
          <span>{t(lang, 'maxSpotsLabel')}</span>
          <input
            type="number"
            min={1}
            max={12}
            value={form.maxSpots}
            onChange={(e) => onFieldChange('maxSpots', Number(e.target.value))}
            disabled={isBusy}
          />
        </label>

        <label className="field field-full">
          <span>{t(lang, 'transportModeLabel')}</span>
          <select
            value={form.transportMode}
            onChange={(e) => onFieldChange('transportMode', e.target.value as TransportMode)}
            disabled={isBusy}
          >
            <option value="walking">{t(lang, 'transportModeWalking')}</option>
            <option value="transit">{t(lang, 'transportModeTransit')}</option>
            <option value="driving">{t(lang, 'transportModeDriving')}</option>
          </select>
        </label>
      </div>

      {isRouteMode && !form.origin.trim() && (
        <p className="inline-note">{t(lang, 'requiredForRoute')}</p>
      )}

      {isRouteMode && (
        <div className="candidate-block">
          <div className="candidate-block-header">
            <div>
              <h3>{t(lang, 'candidateTitle')}</h3>
              <p>{candidatesReady ? t(lang, 'candidateHint') : t(lang, 'noCandidates')}</p>
            </div>
            <button
              className="ghost-button"
              onClick={onGenerateCandidates}
              disabled={isBusy || !form.destination.trim() || !form.origin.trim()}
              type="button"
            >
              {candidatesReady ? t(lang, 'refreshCandidates') : t(lang, 'generateCandidates')}
            </button>
          </div>

          <div className="candidate-list">
            {candidates.map((candidate) => {
              const selected = selectedCandidateIds.includes(candidate.id)
              return (
                <button
                  key={candidate.id}
                  type="button"
                  className={`candidate-card ${selected ? 'selected' : ''}`}
                  onClick={() => onToggleCandidate(candidate.id)}
                >
                  <div className="candidate-card-top">
                    <h4>{candidate.name}</h4>
                    <span>{candidate.role}</span>
                  </div>
                  <p>{candidate.reason}</p>
                </button>
              )
            })}
          </div>
        </div>
      )}

      <div className="panel-actions">
        {isRouteMode ? (
          <button
            className="primary-button"
            type="button"
            onClick={onStartPlanning}
            disabled={!canStartPlanning}
          >
            {t(lang, 'confirmAndStart')}
          </button>
        ) : (
          <button
            className="primary-button"
            type="button"
            onClick={onStartPlanning}
            disabled={!canStartPlanning}
          >
            {t(lang, 'startPlanning')}
          </button>
        )}
      </div>
    </section>
  )
}
