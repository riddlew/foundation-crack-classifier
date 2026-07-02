import { useState } from 'react'
import { useNavigate } from '@tanstack/react-router'
import type { ClassifyResponse } from './classifier'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export function useClassifier() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  async function classify(files: File[], notes: string): Promise<void> {
    setLoading(true)
    setError(null)

    const formData = new FormData()
    for (const file of files) {
      formData.append('files', file)
    }
    const trimmed = notes.trim()
    if (trimmed) {
      formData.append('notes', trimmed)
    }

    try {
      const response = await fetch(`${API_BASE_URL}/classify`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        setError(`Server error: ${response.status}`)
        return
      }

      const data: ClassifyResponse = await response.json()
      await navigate({ to: '/results', state: { results: data, notes: trimmed } })
    } catch {
      setError('Network error — could not reach the classifier service.')
    } finally {
      setLoading(false)
    }
  }

  return { loading, error, classify, clearError: () => setError(null) }
}
