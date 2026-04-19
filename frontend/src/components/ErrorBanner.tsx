import type { Lang } from '../i18n'
import { t } from '../i18n'
import type { ErrorInfo } from '../types'

interface Props {
  error: ErrorInfo
  lang: Lang
  onRetry: () => void
  onOpenSettings: () => void
}

export default function ErrorBanner({ error, lang, onRetry, onOpenSettings }: Props) {
  const needsSettings = error.field === 'amap_api_key' || error.field === 'dashscope_api_key'
  const isDistanceLimit = error.type === 'distance_limit_error'

  return (
    <section className="error-banner">
      <div>
        <p className="error-title">{t(lang, 'errorPrefix')}</p>
        <p className="error-message">{error.message}</p>
        {isDistanceLimit && (
          <p className="error-hint">{t(lang, 'distanceLimitHint')}</p>
        )}
      </div>
      <div className="error-actions">
        {needsSettings && (
          <button className="ghost-button" type="button" onClick={onOpenSettings}>
            {t(lang, 'openSettings')}
          </button>
        )}
        <button className="primary-button subtle" type="button" onClick={onRetry}>
          {t(lang, 'retry')}
        </button>
      </div>
    </section>
  )
}
