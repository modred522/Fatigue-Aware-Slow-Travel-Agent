import type { ErrorInfo } from '../types'
import type { Lang } from '../i18n'
import { t } from '../i18n'

interface Props {
  error: ErrorInfo
  lang: Lang
  onRetry: () => void
  onOpenSettings: () => void
}

export default function ErrorBanner({ error, lang, onRetry, onOpenSettings }: Props) {
  const isConfigError = error.type === 'config_error' || error.type === 'api_key_error'

  return (
    <div className={`error-banner error-${error.type}`}>
      <div className="error-content">
        <div className="error-type">{t(lang, 'errorPrefix')}</div>
        <div className="error-message">{error.message}</div>
      </div>
      <div className="error-actions">
        {isConfigError && (
          <button className="btn-link" onClick={onOpenSettings}>
            ⚙️ {t(lang, 'goToSettings')}
          </button>
        )}
        <button className="btn-link" onClick={onRetry}>
          🔄 {t(lang, 'retryButton')}
        </button>
      </div>
    </div>
  )
}
