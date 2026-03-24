import type { StepEvent } from '../types'
import type { Lang } from '../i18n'
import { t } from '../i18n'

interface Props {
  steps: StepEvent[]
  lang: Lang
}

export default function ItinerarySummary({ steps, lang }: Props) {
  const itinerary: string[] = []
  let totalDistance = 0

  for (const step of steps) {
    if (step.updates.itinerary) {
      itinerary.push(...step.updates.itinerary)
    }
    if (step.updates.cumulative_distance) {
      totalDistance += step.updates.cumulative_distance
    }
  }

  if (itinerary.length === 0) return null

  return (
    <div className="itinerary-summary">
      <h3>🏁 {t(lang, 'finalItinerary')}</h3>
      <ol>
        {itinerary.map((spot, i) => (
          <li key={i} className={spot.startsWith('[Rest Stop]') ? 'rest-item' : ''}>
            {spot}
          </li>
        ))}
      </ol>
      <div className="total-distance">
        🚶 {t(lang, 'totalDistance')}：<strong>{totalDistance}</strong> {t(lang, 'meters')}
      </div>
    </div>
  )
}
