import { useState, useEffect } from 'react'
import type { Lang } from '../i18n'
import { t } from '../i18n'
import type { SettingsData } from '../types'

interface Props {
  open: boolean
  onClose: () => void
  lang: Lang
}

const MODEL_OPTIONS = [
  'qwen-turbo',
  'qwen-plus',
  'qwen-max',
]

export default function Settings({ open, onClose, lang }: Props) {
  const [settings, setSettings] = useState<SettingsData | null>(null)
  const [amapKey, setAmapKey] = useState('')
  const [dashscopeKey, setDashscopeKey] = useState('')
  const [llmModel, setLlmModel] = useState('qwen-turbo')
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved'>('idle')
  const [error, setError] = useState('')

  useEffect(() => {
    if (open) {
      fetch('/api/settings')
        .then((r) => r.json())
        .then((data: SettingsData) => {
          setSettings(data)
          setLlmModel(data.llm_model)
          setAmapKey('')
          setDashscopeKey('')
          setError('')
          setSaveStatus('idle')
        })
        .catch(() => setError('Failed to load settings'))
    }
  }, [open])

  if (!open) return null

  const handleSave = async () => {
    setSaveStatus('saving')
    setError('')
    try {
      const payload: Record<string, string> = {}
      if (amapKey) payload.amap_api_key = amapKey
      if (dashscopeKey) payload.dashscope_api_key = dashscopeKey
      payload.llm_model = llmModel

      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error('Save failed')
      setSaveStatus('saved')
      // Refresh settings display
      const updated = await fetch('/api/settings').then((r) => r.json())
      setSettings(updated)
      setAmapKey('')
      setDashscopeKey('')
      setTimeout(() => setSaveStatus('idle'), 2000)
    } catch {
      setError('Failed to save settings')
      setSaveStatus('idle')
    }
  }

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>{t(lang, 'settingsTitle')}</h2>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="settings-body">
          {/* AMap API Key */}
          <div className="setting-group">
            <label>{t(lang, 'amapKey')}</label>
            <div className="setting-status">
              {settings?.amap_api_key_set ? (
                <span className="status-badge configured">{t(lang, 'configured')}: {settings.amap_api_key}</span>
              ) : (
                <span className="status-badge not-configured">{t(lang, 'notConfigured')}</span>
              )}
            </div>
            <input
              type="text"
              value={amapKey}
              onChange={(e) => setAmapKey(e.target.value)}
              placeholder={t(lang, 'amapKeyPlaceholder')}
            />
          </div>

          {/* DashScope API Key */}
          <div className="setting-group">
            <label>{t(lang, 'dashscopeKey')}</label>
            <div className="setting-status">
              {settings?.dashscope_api_key_set ? (
                <span className="status-badge configured">{t(lang, 'configured')}: {settings.dashscope_api_key}</span>
              ) : (
                <span className="status-badge not-configured">{t(lang, 'notConfigured')}</span>
              )}
            </div>
            <input
              type="text"
              value={dashscopeKey}
              onChange={(e) => setDashscopeKey(e.target.value)}
              placeholder={t(lang, 'dashscopeKeyPlaceholder')}
            />
          </div>

          {/* LLM Model */}
          <div className="setting-group">
            <label>{t(lang, 'llmModel')}</label>
            <select value={llmModel} onChange={(e) => setLlmModel(e.target.value)}>
              {MODEL_OPTIONS.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          {error && <div className="settings-error">{error}</div>}
        </div>

        <div className="settings-footer">
          <button className="btn-secondary" onClick={onClose}>{t(lang, 'cancel')}</button>
          <button className="btn-primary" onClick={handleSave} disabled={saveStatus === 'saving'}>
            {saveStatus === 'saving' ? t(lang, 'saving') : saveStatus === 'saved' ? `✓ ${t(lang, 'saved')}` : t(lang, 'save')}
          </button>
        </div>
      </div>
    </div>
  )
}
