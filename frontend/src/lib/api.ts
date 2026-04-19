import type {
  CandidateRequest,
  CandidateWaypoint,
  LocationValidationRequest,
  LocationValidationResponse,
  ModelOption,
  PlanRequest,
  SettingsData,
  StreamEvent,
} from '../types'

interface StreamHandlers {
  signal?: AbortSignal
  onEvent: (event: StreamEvent) => void
}

export async function fetchSettings(): Promise<SettingsData> {
  const response = await fetch('/api/settings')
  if (!response.ok) {
    throw new Error('Failed to load settings')
  }
  return response.json()
}

export async function saveSettings(payload: Record<string, string>): Promise<void> {
  const response = await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw new Error('Failed to save settings')
  }
}

export async function fetchModelOptions(payload: {
  llm_api_key: string
  llm_base_url: string
}): Promise<ModelOption[]> {
  const response = await fetch('/api/settings/models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data?.detail ?? data?.message ?? 'Failed to fetch models')
  }
  return data.models as ModelOption[]
}

export async function fetchCandidates(payload: CandidateRequest): Promise<CandidateWaypoint[]> {
  const response = await fetch('/api/plans/candidates', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data?.detail ?? 'Failed to generate candidates')
  }
  return data.candidates as CandidateWaypoint[]
}

export async function validateLocation(payload: LocationValidationRequest): Promise<LocationValidationResponse> {
  const response = await fetch('/api/validate-location', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data?.detail ?? data?.message ?? 'Failed to validate location')
  }
  return data as LocationValidationResponse
}

export async function streamPlan(payload: PlanRequest, handlers: StreamHandlers): Promise<void> {
  const response = await fetch('/api/plans/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: handlers.signal,
  })

  if (!response.ok || !response.body) {
    const text = await response.text()
    throw new Error(text || 'Failed to start plan stream')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split(/\r?\n\r?\n/)
    buffer = parts.pop() ?? ''

    for (const rawEvent of parts) {
      const parsed = parseSseEvent(rawEvent)
      if (parsed) handlers.onEvent(parsed)
    }
  }

  if (buffer.trim()) {
    const parsed = parseSseEvent(buffer)
    if (parsed) handlers.onEvent(parsed)
  }
}

function parseSseEvent(chunk: string): StreamEvent | null {
  let event = 'message'
  const dataLines: string[] = []

  for (const line of chunk.split(/\r?\n/)) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim())
    }
  }

  if (!dataLines.length) return null

  return {
    event,
    payload: JSON.parse(dataLines.join('\n')) as Record<string, unknown>,
  }
}
