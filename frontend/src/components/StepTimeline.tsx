import type { Lang } from '../i18n'
import { t } from '../i18n'
import type { StreamEvent } from '../types'

interface Props {
  events: StreamEvent[]
  lang: Lang
}

export default function StepTimeline({ events, lang }: Props) {
  return (
    <section className="panel-card timeline-card">
      <div className="panel-header compact">
        <div>
          <p className="eyebrow">{t(lang, 'dashboard')}</p>
          <h2>{t(lang, 'executionLog')}</h2>
        </div>
      </div>

      {!events.length && <p className="empty-state">{t(lang, 'noTimeline')}</p>}

      <div className="timeline-list">
        {events.map((event, index) => (
          <article key={`${event.event}-${index}`} className="timeline-item">
            <div className="timeline-badge">{labelForEvent(event.event, lang)}</div>
            <div className="timeline-content">
              <h3>{labelForEvent(event.event, lang)}</h3>
              <p>{describeEvent(event)}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

function labelForEvent(event: string, lang: Lang) {
  switch (event) {
    case 'planning_started':
      return t(lang, 'planningStarted')
    case 'itinerary_item_added':
      return t(lang, 'itineraryAdded')
    case 'segment_distance_updated':
      return t(lang, 'segmentUpdated')
    case 'fatigue_status_updated':
      return t(lang, 'fatigueUpdated')
    case 'rest_stop_added':
      return t(lang, 'restStopAdded')
    case 'planning_completed':
      return t(lang, 'completed')
    case 'candidate_set_ready':
      return t(lang, 'candidateReady')
    case 'candidate_generated':
      return t(lang, 'candidateGenerated')
    default:
      return event
  }
}

function describeEvent(event: StreamEvent) {
  if (event.event === 'planning_started') {
    return `mode=${String(event.payload.mode)} destination=${String(event.payload.destination)}`
  }
  if (event.event === 'itinerary_item_added') {
    const item = event.payload.item as { name?: string; reason?: string } | undefined
    return `${item?.name ?? 'Unknown'} · ${item?.reason ?? ''}`.trim()
  }
  if (event.event === 'segment_distance_updated') {
    return `${String(event.payload.from)} → ${String(event.payload.to)} · ${String(event.payload.segment_distance_meters)} m`
  }
  if (event.event === 'fatigue_status_updated') {
    return `cumulative=${String(event.payload.cumulative_distance_meters)} m · threshold=${String(event.payload.fatigue_threshold_meters)} m`
  }
  if (event.event === 'rest_stop_added') {
    const item = event.payload.item as { name?: string; reason?: string } | undefined
    return `${item?.name ?? 'Unknown'} · ${item?.reason ?? ''}`.trim()
  }
  if (event.event === 'planning_completed') {
    return `${String(event.payload.total_distance_meters)} m · ${String(event.payload.rest_stop_count)} rest stops`
  }
  if (event.event === 'candidate_set_ready') {
    return String(event.payload.description ?? '')
  }
  if (event.event === 'candidate_generated') {
    return `${String(event.payload.name)} · ${String(event.payload.role)}`
  }
  return JSON.stringify(event.payload)
}
