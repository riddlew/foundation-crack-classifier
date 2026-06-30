import { ref } from 'vue'

export interface ClassifierResult {
  severity_level: string
  urgency: string
  final_label: string
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

export const classifyResults = ref<ClassifyResponse | null>(null)
export const classifyNotes = ref<string>('')
