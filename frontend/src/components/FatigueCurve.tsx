import { useMemo } from 'react'

import type { Lang } from '../i18n'
import { t } from '../i18n'
import type { ItineraryItem, TravelSummary } from '../types'

interface Props {
  summary: TravelSummary | null
  thresholdMeters: number
  lang: Lang
}

interface CurvePoint {
  label: string
  fatigue: number
  kind: ItineraryItem['kind'] | 'reset' | 'start'
}

const CHART_WIDTH = 420
const CHART_HEIGHT = 190
const PADDING_X = 18
const PADDING_Y = 18

export default function FatigueCurve({ summary, thresholdMeters, lang }: Props) {
  const points = useMemo(() => buildCurvePoints(summary), [summary])

  const chart = useMemo(() => {
    if (!points.length) return null

    const maxFatigue = Math.max(thresholdMeters, ...points.map((point) => point.fatigue), 1)
    const usableWidth = CHART_WIDTH - PADDING_X * 2
    const usableHeight = CHART_HEIGHT - PADDING_Y * 2
    const stepCount = Math.max(points.length - 1, 1)

    const svgPoints = points.map((point, index) => {
      const x = PADDING_X + (usableWidth * index) / stepCount
      const y = CHART_HEIGHT - PADDING_Y - (usableHeight * point.fatigue) / maxFatigue
      return { ...point, x, y }
    })

    const linePath = svgPoints.map((point) => `${point.x},${point.y}`).join(' ')
    const thresholdY = CHART_HEIGHT - PADDING_Y - (usableHeight * thresholdMeters) / maxFatigue

    return { svgPoints, linePath, thresholdY, maxFatigue }
  }, [points, thresholdMeters])

  const stats = useMemo(() => {
    const peak = points.reduce((max, point) => Math.max(max, point.fatigue), 0)
    const resets = points.filter((point) => point.kind === 'reset').length
    const endFatigue = points.length ? points[points.length - 1].fatigue : 0
    return { peak, resets, endFatigue }
  }, [points])

  return (
    <article className="panel-card fatigue-card">
      <p className="eyebrow">{t(lang, 'dashboard')}</p>
      <div className="fatigue-header">
        <div>
          <h2>{t(lang, 'fatiguePanel')}</h2>
          <p>{localText(lang, 'desc')}</p>
        </div>
        <div className="fatigue-legend">
          <span className="legend-chip threshold">{t(lang, 'fatigueThreshold')}</span>
          <span className="legend-chip route">{localText(lang, 'route')}</span>
        </div>
      </div>

      {!chart && <p className="empty-state">{localText(lang, 'empty')}</p>}

      {chart && (
        <>
          <div className="fatigue-chart-shell">
            <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} className="fatigue-chart" role="img" aria-label={t(lang, 'fatiguePanel')}>
              <line
                x1={PADDING_X}
                x2={CHART_WIDTH - PADDING_X}
                y1={chart.thresholdY}
                y2={chart.thresholdY}
                className="fatigue-threshold-line"
              />
              <polyline points={chart.linePath} className="fatigue-route-line" />
              {chart.svgPoints.map((point, index) => (
                <g key={`${point.label}-${index}`}>
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r={point.kind === 'reset' ? 4.5 : 4}
                    className={`fatigue-point point-${point.kind}`}
                  />
                </g>
              ))}
            </svg>

            <div className="fatigue-x-axis">
              {chart.svgPoints.map((point, index) => (
                <div key={`${point.label}-${index}`} className="fatigue-axis-label">
                  <span>{shortLabel(point.label)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="fatigue-stats-grid">
            <div className="metric-card">
              <span>{localText(lang, 'peak')}</span>
              <strong>{stats.peak} m</strong>
            </div>
            <div className="metric-card">
              <span>{localText(lang, 'resets')}</span>
              <strong>{stats.resets}</strong>
            </div>
            <div className="metric-card">
              <span>{localText(lang, 'ending')}</span>
              <strong>{stats.endFatigue} m</strong>
            </div>
          </div>
        </>
      )}
    </article>
  )
}

function buildCurvePoints(summary: TravelSummary | null): CurvePoint[] {
  if (!summary) return []

  const points: CurvePoint[] = [{ label: 'Start', fatigue: 0, kind: 'start' }]

  for (const item of summary.itinerary_items) {
    if (item.kind === 'origin') continue

    if (item.kind === 'rest_stop') {
      const arrivalFatigue = Math.max(item.cumulative_distance_meters, item.distance_from_previous_meters)
      points.push({
        label: item.name,
        fatigue: arrivalFatigue,
        kind: 'rest_stop',
      })
      points.push({
        label: `${item.name} reset`,
        fatigue: 0,
        kind: 'reset',
      })
      continue
    }

    points.push({
      label: item.name,
      fatigue: item.cumulative_distance_meters,
      kind: item.kind,
    })
  }

  return points
}

function shortLabel(label: string): string {
  return label.length > 12 ? `${label.slice(0, 10)}..` : label
}

function localText(
  lang: Lang,
  key: 'desc' | 'route' | 'empty' | 'peak' | 'resets' | 'ending',
): string {
  const zh = {
    desc: '按 itinerary 段落展示疲劳累积与休息重置。',
    route: '行走负载',
    empty: '规划完成后，这里会显示疲劳曲线。',
    peak: '峰值疲劳',
    resets: '重置次数',
    ending: '末段疲劳',
  }
  const en = {
    desc: 'Tracks fatigue buildup and reset points across the itinerary.',
    route: 'Route Load',
    empty: 'The fatigue curve will appear here after planning completes.',
    peak: 'Peak fatigue',
    resets: 'Reset count',
    ending: 'Ending fatigue',
  }
  return (lang === 'zh' ? zh : en)[key]
}
