import type { StepEvent } from '../types'
import type { Lang } from '../i18n'
import { t } from '../i18n'

interface Props {
  steps: StepEvent[]
  lang: Lang
}

const nodeIcons: Record<string, string> = {
  planner: '🧠',
  calculator: '📏',
  rest_stop: '☕',
}

export default function StepTimeline({ steps, lang }: Props) {
  if (steps.length === 0) return null

  const nodeLabels: Record<string, string> = {
    planner: t(lang, 'plannerLabel'),
    calculator: t(lang, 'calculatorLabel'),
    rest_stop: t(lang, 'restStopLabel'),
  }

  return (
    <div className="step-timeline">
      <h3>{t(lang, 'process')}</h3>
      {steps.map((step, index) => {
        const icon = nodeIcons[step.node] || '⚙️'
        const label = nodeLabels[step.node] || step.node
        return (
          <div key={index} className={`step-card step-${step.node}`}>
            <div className="step-header">
              <span className="step-icon">{icon}</span>
              <span className="step-label">{label}</span>
            </div>
            <div className="step-body">
              {renderStepContent(step, lang)}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function renderStepContent(step: StepEvent, lang: Lang) {
  const { node, updates } = step

  if (node === 'planner' && updates.itinerary?.length) {
    return <p>{t(lang, 'foundSpot')}：<strong>{updates.itinerary[0]}</strong></p>
  }

  if (node === 'calculator') {
    const dist = updates.cumulative_distance ?? 0
    const needsRest = updates.needs_rest
    return (
      <>
        <p>{t(lang, 'segmentDistance')}：<strong>{dist}{t(lang, 'meters')}</strong></p>
        {needsRest !== undefined && (
          <p className={needsRest ? 'warning' : 'ok'}>
            {needsRest ? `⚠️ ${t(lang, 'fatigueWarning')}` : `✅ ${t(lang, 'energyOk')}`}
          </p>
        )}
      </>
    )
  }

  if (node === 'rest_stop' && updates.itinerary?.length) {
    return <p>{t(lang, 'restRecommend')}：<strong>{updates.itinerary[0]}</strong></p>
  }

  return <p>{JSON.stringify(updates)}</p>
}
