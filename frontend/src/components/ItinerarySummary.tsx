import type { Lang } from '../i18n'
import { t } from '../i18n'
import type { TravelSummary } from '../types'

interface Props {
  summary: TravelSummary | null
  lang: Lang
}

export default function ItinerarySummary({ summary, lang }: Props) {
  return (
    <section className="panel-card summary-card">
      <div className="panel-header compact">
        <div>
          <p className="eyebrow">{t(lang, 'dashboard')}</p>
          <h2>{t(lang, 'summary')}</h2>
        </div>
      </div>

      {!summary && <p className="empty-state">{t(lang, 'noSummary')}</p>}

      {summary && (
        <>
          <div className="summary-metrics">
            <div className="metric-card">
              <span>{t(lang, 'totalDistance')}</span>
              <strong>{summary.total_distance_meters} m</strong>
            </div>
            <div className="metric-card">
              <span>{t(lang, 'fatigueThreshold')}</span>
              <strong>{summary.fatigue_threshold_meters} m</strong>
            </div>
            <div className="metric-card">
              <span>{t(lang, 'restStops')}</span>
              <strong>{summary.rest_stop_count}</strong>
            </div>
          </div>

          <ol className="itinerary-list">
            {summary.itinerary_items.map((item) => (
              <li key={item.id}>
                <div className="itinerary-row">
                  <div>
                    <p className="itinerary-name">{item.name}</p>
                    <p className="itinerary-reason">{item.reason}</p>
                  </div>
                  <span className={`kind-tag kind-${item.kind}`}>{item.kind}</span>
                </div>
                <div className="itinerary-meta">
                  <span>{item.distance_from_previous_meters} m</span>
                  <span>{item.cumulative_distance_meters} m</span>
                </div>
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  )
}
