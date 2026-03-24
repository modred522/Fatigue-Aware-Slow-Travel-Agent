export interface StepEvent {
  node: 'planner' | 'calculator' | 'rest_stop'
  updates: {
    itinerary?: string[]
    cumulative_distance?: number
    current_location?: string
    needs_rest?: boolean
  }
}

export type AppStatus = 'idle' | 'running' | 'done' | 'error'

export interface ErrorInfo {
  type: string
  message: string
  field?: string
}

export interface SettingsData {
  amap_api_key: string
  amap_api_key_set: boolean
  dashscope_api_key: string
  dashscope_api_key_set: boolean
  llm_model: string
}
