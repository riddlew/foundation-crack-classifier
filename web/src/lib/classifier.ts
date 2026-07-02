export interface ClassifierResult {
  severity_level: string
  urgency: 'inspection_recommended' | 'contact_soon' | 'contact_immediately' | 'unable_to_assess'
  final_label: 'level1' | 'level2' | 'level3' | 'unclear'
  confidence: number
  raw_probabilities: Record<string, number>
  why_this_result: string
  customer_summary: string
  disclaimer: string
  recommended_action: string
}

export interface FileClassificationResponse {
  filename: string
  ok: boolean
  result: ClassifierResult | null
  error: string | null
}

export interface ClassifyResponse {
  results: FileClassificationResponse[]
}

// The upload page hands results to the results page through router navigation
// state, so teach the router what that state carries.
declare module '@tanstack/history' {
  interface HistoryState {
    results?: ClassifyResponse
    notes?: string
  }
}
