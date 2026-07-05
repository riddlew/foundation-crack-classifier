import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useClassifier } from '../useClassifier'

const { mockNavigate, mockSetBatch } = vi.hoisted(() => ({
  mockNavigate: vi.fn(),
  mockSetBatch: vi.fn(() => 'test-batch-id'),
}))

vi.mock('@tanstack/react-router', () => ({
  useNavigate: () => mockNavigate,
}))

vi.mock('../imageStore', () => ({
  setBatch: mockSetBatch,
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

function fetchMock() {
  return globalThis.fetch as ReturnType<typeof vi.fn>
}

beforeEach(() => {
  mockNavigate.mockReset()
  mockSetBatch.mockClear()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useClassifier', () => {
  it('POSTs files and notes to /classify then navigates to /results with state', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(FAKE_RESPONSE) }),
    )

    const { result } = renderHook(() => useClassifier())
    const file = new File(['bytes'], 'test.jpg', { type: 'image/jpeg' })

    await act(async () => {
      await result.current.classify([file], 'image #1 shows the east wall')
    })

    const [url, init] = fetchMock().mock.calls[0]
    expect(url).toBe('http://localhost:8000/classify')
    expect(init.method).toBe('POST')
    expect((init.body as FormData).get('notes')).toBe('image #1 shows the east wall')
    expect(mockSetBatch).toHaveBeenCalledWith([file])
    expect(mockNavigate).toHaveBeenCalledWith({
      to: '/results',
      state: {
        results: FAKE_RESPONSE,
        notes: 'image #1 shows the east wall',
        imageBatchId: 'test-batch-id',
      },
    })
    expect(result.current.loading).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('omits notes from FormData when notes is blank', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ results: [] }) }),
    )

    const { result } = renderHook(() => useClassifier())
    await act(async () => {
      await result.current.classify([new File(['b'], 'x.jpg')], '   ')
    })

    const body = fetchMock().mock.calls[0][1].body as FormData
    expect(body.has('notes')).toBe(false)
  })

  it('passes an empty notes string in navigation state when notes is blank', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ results: [] }) }),
    )

    const { result } = renderHook(() => useClassifier())
    await act(async () => {
      await result.current.classify([new File(['b'], 'x.jpg')], '   ')
    })

    expect(mockNavigate).toHaveBeenCalledWith({
      to: '/results',
      state: { results: { results: [] }, notes: '', imageBatchId: 'test-batch-id' },
    })
  })

  it('sets error on network failure without navigating', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Failed to fetch')))

    const { result } = renderHook(() => useClassifier())
    await act(async () => {
      await result.current.classify([new File(['b'], 'x.jpg')], '')
    })

    expect(result.current.error).toBe('Network error — could not reach the classifier service.')
    expect(mockNavigate).not.toHaveBeenCalled()
  })

  it('sets error on non-2xx response without navigating', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }))

    const { result } = renderHook(() => useClassifier())
    await act(async () => {
      await result.current.classify([new File(['b'], 'x.jpg')], '')
    })

    expect(result.current.error).toBe('Server error: 500')
    expect(mockNavigate).not.toHaveBeenCalled()
  })

  it('sets loading to true while the fetch is in-flight', async () => {
    let resolveFetch!: (v: unknown) => void
    vi.stubGlobal(
      'fetch',
      vi.fn().mockReturnValue(new Promise((r) => { resolveFetch = r })),
    )

    const { result } = renderHook(() => useClassifier())
    let promise!: Promise<void>
    act(() => {
      promise = result.current.classify([new File(['b'], 'x.jpg')], '')
    })
    expect(result.current.loading).toBe(true)

    await act(async () => {
      resolveFetch({ ok: true, json: () => Promise.resolve({ results: [] }) })
      await promise
    })
    expect(result.current.loading).toBe(false)
  })

  it('resets loading to false after network failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Failed to fetch')))
    const { result } = renderHook(() => useClassifier())
    await act(async () => {
      await result.current.classify([new File(['b'], 'x.jpg')], '')
    })
    expect(result.current.loading).toBe(false)
  })

  it('appends all files to FormData', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ results: [] }) }),
    )
    const { result } = renderHook(() => useClassifier())
    const files = [new File(['a'], 'a.jpg'), new File(['b'], 'b.jpg')]
    await act(async () => {
      await result.current.classify(files, '')
    })
    const body = fetchMock().mock.calls[0][1].body as FormData
    expect(body.getAll('files')).toHaveLength(2)
  })
})
