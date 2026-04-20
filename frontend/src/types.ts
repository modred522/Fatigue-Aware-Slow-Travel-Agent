export type PlanningMode = 'local_explore' | 'point_to_point'

export type TransportMode = 'walking' | 'transit' | 'driving'

export type AppStatus =
  | 'idle'
  | 'loading_candidates'
  | 'candidate_ready'
  | 'running'
  | 'completed'
  | 'error'

export interface ErrorInfo {
  type: string
  message: string
  field?: string
}

export interface SettingsData {
  amap_api_key: string
  amap_api_key_set: boolean
  llm_api_key: string
  llm_api_key_set: boolean
  llm_base_url: string
  llm_model: string
  llm_temperature: number
}

export interface ModelOption {
  id: string
}

export interface CandidateWaypoint {
  id: string
  name: string
  reason: string
  role: string
}

export interface PlanningFormState {
  city: string
  destination: string
  origin: string
  tripDurationDays: number
  interestsText: string
  fatigueThresholdMeters: number
  maxSpots: number
  transportMode: TransportMode
}

export interface CandidateRequest {
  mode: 'point_to_point'
  origin: string
  destination: string
  interests: string[]
  trip_duration_days: number
}

export interface PlanRequest {
  mode: PlanningMode
  city: string
  destination: string
  origin: string
  trip_duration_days: number
  interests: string[]
  fatigue_threshold_meters: number
  max_spots: number
  selected_waypoints: CandidateWaypoint[]
  transport_mode: TransportMode
}

export interface ItineraryItem {
  id: string
  name: string
  kind: 'origin' | 'spot' | 'waypoint' | 'rest_stop' | 'destination'
  sequence: number
  anchor_segment: string
  distance_from_previous_meters: number
  cumulative_distance_meters: number
  reason: string
  confirmed: boolean
  transport_mode?: TransportMode
  location_coords?: string | null
}

export interface MapSettings {
  amap_api_key: string
  amap_api_key_set: boolean
}

export interface TravelSummary {
  mode: PlanningMode
  origin: string
  destination: string
  total_distance_meters: number
  fatigue_threshold_meters: number
  rest_stop_count: number
  itinerary_items: ItineraryItem[]
  transport_mode: TransportMode
}

export interface StreamEvent {
  event: string
  payload: Record<string, unknown>
}

export interface LocationValidationRequest {
  city: string
  location: string
}

export interface LocationValidationResponse {
  valid: boolean
  city: string
  location: string
  formatted_address?: string
  message?: string
}
