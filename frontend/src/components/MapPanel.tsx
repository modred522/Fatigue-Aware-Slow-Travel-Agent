import { useEffect, useMemo, useRef, useState } from 'react'
import type { MutableRefObject } from 'react'

import type { Lang } from '../i18n'
import { t } from '../i18n'
import { fetchMapSettings } from '../lib/api'
import type { ItineraryItem, MapSettings, TravelSummary } from '../types'

interface Props {
  summary: TravelSummary | null
  lang: Lang
}

interface MapPoint {
  name: string
  kind: ItineraryItem['kind']
  lng: number
  lat: number
}

declare global {
  interface Window {
    AMap?: any
  }
}

let amapScriptPromise: Promise<any> | null = null

export default function MapPanel({ summary, lang }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<any>(null)
  const overlaysRef = useRef<any[]>([])
  const [mapSettings, setMapSettings] = useState<MapSettings | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  const points = useMemo(() => buildMapPoints(summary), [summary])

  useEffect(() => {
    void fetchMapSettings()
      .then(setMapSettings)
      .catch(() => setLoadError(localText(lang, 'settingsError')))
  }, [lang])

  useEffect(() => {
    if (!containerRef.current || !mapSettings?.amap_api_key) return

    let disposed = false

    loadAmapSdk(mapSettings.amap_api_key)
      .then((AMap) => {
        if (disposed || !containerRef.current) return
        if (!mapRef.current) {
          mapRef.current = new AMap.Map(containerRef.current, {
            viewMode: '2D',
            zoom: 12,
            mapStyle: 'amap://styles/whitesmoke',
            resizeEnable: true,
          })
        }

        renderMap(AMap, mapRef.current, overlaysRef, points)
      })
      .catch(() => setLoadError(localText(lang, 'loadError')))

    return () => {
      disposed = true
    }
  }, [mapSettings, points, lang])

  return (
    <article className="panel-card map-card">
      <p className="eyebrow">{t(lang, 'dashboard')}</p>
      <div className="map-panel-header">
        <div>
          <h2>{t(lang, 'mapPlaceholder')}</h2>
          <p>{localText(lang, 'desc')}</p>
        </div>
        <div className="map-status-chip">{points.length ? `${points.length} pts` : localText(lang, 'emptyTag')}</div>
      </div>

      {!mapSettings?.amap_api_key_set && <p className="empty-state">{localText(lang, 'missingKey')}</p>}
      {loadError && <p className="empty-state">{loadError}</p>}
      {mapSettings?.amap_api_key_set && !points.length && !loadError && (
        <p className="empty-state">{localText(lang, 'waiting')}</p>
      )}

      <div className={`map-canvas ${!mapSettings?.amap_api_key_set || loadError ? 'hidden' : ''}`} ref={containerRef} />
    </article>
  )
}

function buildMapPoints(summary: TravelSummary | null): MapPoint[] {
  if (!summary) return []

  return summary.itinerary_items
    .filter((item) => Boolean(item.location_coords))
    .map((item) => {
      const [lng, lat] = String(item.location_coords).split(',').map(Number)
      return {
        name: item.name,
        kind: item.kind,
        lng,
        lat,
      }
    })
    .filter((point) => Number.isFinite(point.lng) && Number.isFinite(point.lat))
}

function loadAmapSdk(apiKey: string): Promise<any> {
  if (window.AMap) return Promise.resolve(window.AMap)
  if (amapScriptPromise) return amapScriptPromise

  amapScriptPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>('script[data-amap-sdk="true"]')
    if (existing) {
      existing.addEventListener('load', () => resolve(window.AMap))
      existing.addEventListener('error', reject)
      return
    }

    const script = document.createElement('script')
    script.src = `https://webapi.amap.com/maps?v=2.0&key=${encodeURIComponent(apiKey)}`
    script.async = true
    script.defer = true
    script.dataset.amapSdk = 'true'
    script.onload = () => resolve(window.AMap)
    script.onerror = reject
    document.head.appendChild(script)
  })

  return amapScriptPromise
}

function renderMap(AMap: any, map: any, overlaysRef: MutableRefObject<any[]>, points: MapPoint[]) {
  for (const overlay of overlaysRef.current) {
    map.remove(overlay)
  }
  overlaysRef.current = []

  if (!points.length) return

  const markers = points.map((point, index) => {
    const marker = new AMap.Marker({
      position: [point.lng, point.lat],
      title: point.name,
      label: {
        content: `<div class="map-marker-label">${index + 1}. ${escapeHtml(point.name)}</div>`,
        direction: 'top',
      },
    })
    return marker
  })

  const polyline = new AMap.Polyline({
    path: points.map((point) => [point.lng, point.lat]),
    strokeColor: '#1f8f7a',
    strokeWeight: 5,
    strokeOpacity: 0.85,
    strokeStyle: 'solid',
  })

  map.add([...markers, polyline])
  overlaysRef.current = [...markers, polyline]
  map.setFitView([...markers, polyline], false, [50, 50, 50, 50])
}

function escapeHtml(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

function localText(
  lang: Lang,
  key: 'desc' | 'missingKey' | 'waiting' | 'loadError' | 'settingsError' | 'emptyTag',
): string {
  const zh = {
    desc: '根据 itinerary 坐标展示标记点与路线连线。',
    missingKey: '请先在设置中配置 AMap API Key，地图面板才会启用。',
    waiting: '规划完成后，这里会显示地图标记与路径。',
    loadError: '地图 SDK 加载失败，请检查网络或 AMap key。',
    settingsError: '地图配置读取失败。',
    emptyTag: 'No Data',
  }
  const en = {
    desc: 'Renders itinerary markers and route lines from item coordinates.',
    missingKey: 'Configure the AMap API key in Settings to enable the map panel.',
    waiting: 'Map markers and route lines will appear here after planning completes.',
    loadError: 'Failed to load the map SDK. Check network access or the AMap key.',
    settingsError: 'Failed to load map settings.',
    emptyTag: 'No Data',
  }
  return (lang === 'zh' ? zh : en)[key]
}
