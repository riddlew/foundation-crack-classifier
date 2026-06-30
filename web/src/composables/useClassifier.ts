import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { classifyNotes, classifyResults, type ClassifyResponse } from '../state/results'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export function useClassifier() {
  const loading = ref(false)
  const error = ref<string | null>(null)
  const router = useRouter()

  async function classify(files: File[], notes: string): Promise<void> {
    loading.value = true
    error.value = null

    const formData = new FormData()
    for (const file of files) {
      formData.append('files', file)
    }
    if (notes.trim()) {
      formData.append('notes', notes.trim())
    }

    try {
      const response = await fetch(`${API_BASE_URL}/classify`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        error.value = `Server error: ${response.status}`
        return
      }

      const data: ClassifyResponse = await response.json()
      classifyResults.value = data
      classifyNotes.value = notes.trim()
      router.push('/results')
    } catch {
      error.value = 'Network error — could not reach the classifier service.'
    } finally {
      loading.value = false
    }
  }

  return { loading, error, classify }
}
