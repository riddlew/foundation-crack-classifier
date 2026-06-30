import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useClassifier } from '../useClassifier'
import { classifyNotes, classifyResults } from '../../state/results'

const { mockPush } = vi.hoisted(() => ({ mockPush: vi.fn() }))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: mockPush }),
}))

const FAKE_RESPONSE = {
  results: [
    {
      filename: 'test.jpg',
      ok: true,
      result: {
        severity_level: 'Level 2',
        urgency: 'contact_soon',
        final_label: 'level2',
        confidence: 0.71,
        raw_probabilities: { level1: 0.08, level2: 0.71, level3: 0.13, unclear: 0.08 },
        why_this_result: 'Level 2 threshold crossed.',
        customer_summary: 'May indicate a more concerning crack.',
        disclaimer: 'Not a structural diagnosis.',
        recommended_action: 'Contact a professional soon.',
      },
      error: null,
    },
  ],
}

beforeEach(() => {
  mockPush.mockReset()
  classifyResults.value = null
  classifyNotes.value = ''
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useClassifier', () => {
  it('POSTs files and notes to /classify then navigates to /results', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(FAKE_RESPONSE) }),
    )

    const { classify, loading, error } = useClassifier()
    const file = new File(['bytes'], 'test.jpg', { type: 'image/jpeg' })

    await classify([file], 'image #1 shows the east wall')

    const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('http://localhost:8000/classify')
    expect(init.method).toBe('POST')
    expect((init.body as FormData).get('notes')).toBe('image #1 shows the east wall')
    expect(classifyResults.value).toEqual(FAKE_RESPONSE)
    expect(classifyNotes.value).toBe('image #1 shows the east wall')
    expect(mockPush).toHaveBeenCalledWith('/results')
    expect(loading.value).toBe(false)
    expect(error.value).toBeNull()
  })

  it('omits notes from FormData when notes is blank', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ results: [] }) }),
    )

    const { classify } = useClassifier()
    await classify([new File(['b'], 'x.jpg')], '   ')

    const body = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body as FormData
    expect(body.has('notes')).toBe(false)
  })

  it('sets error on network failure without navigating', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Failed to fetch')))

    const { classify, error } = useClassifier()
    await classify([new File(['b'], 'x.jpg')], '')

    expect(error.value).toBe('Network error — could not reach the classifier service.')
    expect(mockPush).not.toHaveBeenCalled()
    expect(classifyResults.value).toBeNull()
  })

  it('sets error on non-2xx response without navigating', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: false, status: 500 }),
    )

    const { classify, error } = useClassifier()
    await classify([new File(['b'], 'x.jpg')], '')

    expect(error.value).toBe('Server error: 500')
    expect(mockPush).not.toHaveBeenCalled()
  })

  it('sets loading to true while the fetch is in-flight', async () => {
    let resolveJson!: (v: unknown) => void
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => new Promise(r => { resolveJson = r }),
      }),
    )
    const { classify, loading } = useClassifier()
    const promise = classify([new File(['b'], 'x.jpg')], '')
    expect(loading.value).toBe(true)
    await Promise.resolve() // let the fetch microtask settle so json() is called
    resolveJson({ results: [] })
    await promise
    expect(loading.value).toBe(false)
  })

  it('resets loading to false after network failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Failed to fetch')))
    const { classify, loading } = useClassifier()
    await classify([new File(['b'], 'x.jpg')], '')
    expect(loading.value).toBe(false)
  })

  it('resets loading to false after non-2xx response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }))
    const { classify, loading } = useClassifier()
    await classify([new File(['b'], 'x.jpg')], '')
    expect(loading.value).toBe(false)
  })

  it('appends all files to FormData', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ results: [] }) }),
    )
    const { classify } = useClassifier()
    const files = [new File(['a'], 'a.jpg'), new File(['b'], 'b.jpg')]
    await classify(files, '')
    const body = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body as FormData
    expect(body.getAll('files')).toHaveLength(2)
  })
})
