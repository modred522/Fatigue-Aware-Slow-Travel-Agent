import { useEffect, useState } from 'react'

import type { Lang } from '../i18n'
import { t } from '../i18n'
import { fetchModelOptions, fetchSettings, saveSettings } from '../lib/api'
import type { ModelOption, SettingsData } from '../types'

interface Props {
  open: boolean
  onClose: () => void
  lang: Lang
}

export default function Settings({ open, onClose, lang }: Props) {
  const [settings, setSettings] = useState<SettingsData | null>(null)
  const [amapKey, setAmapKey] = useState('')
  const [llmApiKey, setLlmApiKey] = useState('')
  const [llmBaseUrl, setLlmBaseUrl] = useState('')
  const [llmModel, setLlmModel] = useState('qwen-turbo')
  const [llmTemperature, setLlmTemperature] = useState(0.3)
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([])
  const [saving, setSaving] = useState(false)
  const [fetchingModels, setFetchingModels] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    if (!open) return
    void fetchSettings()
      .then((data) => {
        setSettings(data)
        setLlmModel(data.llm_model)
        setLlmTemperature(data.llm_temperature ?? 0.3)
        setAmapKey('')
        setLlmApiKey('')
        setLlmBaseUrl(data.llm_base_url)
        setModelOptions(data.llm_model ? [{ id: data.llm_model }] : [])
        setMessage('')
      })
      .catch(() => setMessage('Failed to load settings'))
  }, [open])

  if (!open) return null

  const handleFetchModels = async () => {
    setFetchingModels(true)
    setMessage('')
    try {
      const models = await fetchModelOptions({
        llm_api_key: llmApiKey.trim() || settings?.llm_api_key || '',
        llm_base_url: llmBaseUrl.trim() || settings?.llm_base_url || '',
      })
      setModelOptions(models)
      if (models.length && !models.some((item) => item.id === llmModel)) {
        setLlmModel(models[0].id)
      }
      setMessage(t(lang, 'modelsFetched'))
    } catch {
      setMessage('Failed to fetch models')
    } finally {
      setFetchingModels(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    setMessage('')
    try {
      const payload: Record<string, string> = { llm_model: llmModel, llm_temperature: String(llmTemperature) }
      if (amapKey.trim()) payload.amap_api_key = amapKey.trim()
      if (llmApiKey.trim()) payload.llm_api_key = llmApiKey.trim()
      payload.llm_base_url = llmBaseUrl.trim()
      await saveSettings(payload)
      const latest = await fetchSettings()
      setSettings(latest)
      setAmapKey('')
      setLlmApiKey('')
      setLlmBaseUrl(latest.llm_base_url)
      setMessage(t(lang, 'saved'))
    } catch {
      setMessage('Failed to save settings')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <div>
            <p className="eyebrow">{t(lang, 'settings')}</p>
            <h2>{t(lang, 'settingsTitle')}</h2>
          </div>
          <button className="close-button" type="button" onClick={onClose}>
            x
          </button>
        </div>

        <div className="settings-body">
          <label className="field">
            <span>{t(lang, 'amapKey')}</span>
            <small>{settings?.amap_api_key_set ? `${t(lang, 'configured')}: ${settings.amap_api_key}` : t(lang, 'notConfigured')}</small>
            <input
              value={amapKey}
              onChange={(e) => setAmapKey(e.target.value)}
              placeholder={t(lang, 'amapKeyPlaceholder')}
            />
          </label>

          <label className="field">
            <span>{t(lang, 'llmApiKey')}</span>
            <small>{settings?.llm_api_key_set ? `${t(lang, 'configured')}: ${settings.llm_api_key}` : t(lang, 'notConfigured')}</small>
            <input
              value={llmApiKey}
              onChange={(e) => setLlmApiKey(e.target.value)}
              placeholder={t(lang, 'llmApiKeyPlaceholder')}
            />
          </label>

          <label className="field">
            <span>{t(lang, 'llmBaseUrl')}</span>
            <small>{t(lang, 'modelHelp')}</small>
            <input
              value={llmBaseUrl}
              onChange={(e) => setLlmBaseUrl(e.target.value)}
              placeholder={t(lang, 'llmBaseUrlPlaceholder')}
            />
          </label>

          <label className="field">
            <span>{t(lang, 'llmModel')}</span>
            <div className="settings-inline-actions">
              <button className="ghost-button" type="button" onClick={handleFetchModels} disabled={fetchingModels}>
                {fetchingModels ? t(lang, 'fetchingModels') : t(lang, 'fetchModels')}
              </button>
            </div>
            <select value={llmModel} onChange={(e) => setLlmModel(e.target.value)}>
              {modelOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.id}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>{t(lang, 'llmTemperature')}</span>
            <small>{t(lang, 'llmTemperatureHelp')}</small>
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={llmTemperature}
              onChange={(e) => setLlmTemperature(Number(e.target.value))}
            />
          </label>

          {message && <p className="settings-message">{message}</p>}
        </div>

        <div className="settings-footer">
          <button className="ghost-button" type="button" onClick={onClose}>
            {t(lang, 'cancel')}
          </button>
          <button className="primary-button" type="button" onClick={handleSave} disabled={saving}>
            {saving ? t(lang, 'saving') : t(lang, 'save')}
          </button>
        </div>
      </div>
    </div>
  )
}
