import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ClassifyResponse } from '../../lib/classifier'

const { mockNavigate, mockGetUrls, state } = vi.hoisted(() => ({
  mockNavigate: vi.fn(),
  mockGetUrls: vi.fn<(id: string) => string[] | null>(() => null),
  state: {
    value: {} as { results?: ClassifyResponse; notes?: string; imageBatchId?: string },
  },
}))

vi.mock('@tanstack/react-router', () => ({
  useNavigate: () => mockNavigate,
  useLocation: (opts?: { select?: (l: { state: unknown }) => unknown }) => {
    const loc = { state: state.value }
    return opts?.select ? opts.select(loc) : loc
  },
}))

vi.mock('../../lib/imageStore', () => ({
  getUrls: mockGetUrls,
}))

import { ResultsPage } from '../ResultsPage'

const validResponse: ClassifyResponse = {
  results: [
    {
      filename: 'crack.jpg',
      ok: true,
      result: {
        severity_level: 'Level 1',
        urgency: 'inspection_recommended',
        final_label: 'level1',
        confidence: 92.5,
        raw_probabilities: {},
        why_this_result: '',
        customer_summary: 'Minor crack',
        disclaimer: 'Not a diagnosis',
        recommended_action: 'Schedule inspection',
      },
      error: null,
    },
  ],
}

beforeEach(() => {
  mockNavigate.mockReset()
  mockGetUrls.mockReset()
  mockGetUrls.mockReturnValue(null)
  state.value = {}
})

describe('results page', () => {
  it('redirects to / when there are no results in navigation state', () => {
    state.value = {}
    render(<ResultsPage />)
    expect(mockNavigate).toHaveBeenCalledWith({ to: '/', replace: true })
  })

  it('does not redirect when results are present', () => {
    state.value = { results: validResponse }
    render(<ResultsPage />)
    expect(mockNavigate).not.toHaveBeenCalled()
  })

  it('renders notes when present', () => {
    state.value = { results: validResponse, notes: 'test notes' }
    render(<ResultsPage />)
    expect(screen.getByText('Your notes: test notes')).toBeInTheDocument()
  })

  it('renders a success card with correct content', () => {
    state.value = { results: validResponse }
    render(<ResultsPage />)
    expect(screen.getByText('Level 1 — Inspection recommended')).toBeInTheDocument()
    expect(screen.getByText('92.5% confidence')).toBeInTheDocument()
    expect(screen.getByText('crack.jpg')).toBeInTheDocument()
    expect(screen.getByText('Minor crack')).toBeInTheDocument()
    expect(screen.getByText('Schedule inspection')).toBeInTheDocument()
  })

  it('renders an error card when ok is false', () => {
    state.value = {
      results: {
        results: [
          {
            filename: 'bad.jpg',
            ok: false,
            result: null,
            error: 'Decode failed',
          },
        ],
      },
    }
    render(<ResultsPage />)
    expect(screen.getByText('Could not process image')).toBeInTheDocument()
    expect(screen.getByText('bad.jpg')).toBeInTheDocument()
    expect(screen.getByText('Decode failed')).toBeInTheDocument()
  })

  it('renders the uploaded image when the batch resolves', () => {
    state.value = { results: validResponse, imageBatchId: 'b1' }
    mockGetUrls.mockReturnValue(['blob:mock/photo-0'])
    render(<ResultsPage />)
    const img = screen.getByRole('img')
    expect(img).toHaveAttribute('src', 'blob:mock/photo-0')
    expect(mockGetUrls).toHaveBeenCalledWith('b1')
  })

  it('renders no image when the batch cannot be resolved', () => {
    state.value = { results: validResponse, imageBatchId: 'stale' }
    mockGetUrls.mockReturnValue(null)
    render(<ResultsPage />)
    expect(screen.queryByRole('img')).toBeNull()
  })

  it('navigates home on back', async () => {
    state.value = { results: validResponse }
    render(<ResultsPage />)
    await userEvent.click(screen.getByRole('button', { name: '← Back' }))
    expect(mockNavigate).toHaveBeenCalledWith({ to: '/' })
  })
})
