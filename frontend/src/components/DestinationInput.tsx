import { useState } from 'react'
import type { Lang } from '../i18n'
import { t } from '../i18n'

interface Props {
  onStart: (destination: string, origin: string) => void
  disabled: boolean
  lang: Lang
}

export default function DestinationInput({ onStart, disabled, lang }: Props) {
  const [destination, setDestination] = useState('')
  const [origin, setOrigin] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmedDest = destination.trim()
    if (trimmedDest) {
      onStart(trimmedDest, origin.trim())
    }
  }

  return (
    <form className="destination-input" onSubmit={handleSubmit}>
      <div className="input-row">
        <div className="input-group">
          <label className="input-label">{lang === 'zh' ? '起点' : 'From'}</label>
          <input
            type="text"
            value={origin}
            onChange={(e) => setOrigin(e.target.value)}
            placeholder={t(lang, 'originPlaceholder')}
            disabled={disabled}
          />
        </div>
        <div className="input-group">
          <label className="input-label">{lang === 'zh' ? '目的地' : 'Destination'}<span className="required">*</span></label>
          <input
            type="text"
            value={destination}
            onChange={(e) => setDestination(e.target.value)}
            placeholder={t(lang, 'destinationPlaceholder')}
            disabled={disabled}
          />
        </div>
      </div>
      <button type="submit" disabled={disabled || !destination.trim()}>
        {disabled ? t(lang, 'runningButton') : t(lang, 'startButton')}
      </button>
    </form>
  )
}
