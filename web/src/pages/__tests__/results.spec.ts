import { mount } from '@vue/test-utils'
import { vi, beforeEach, describe, it, expect } from 'vitest'
import ResultsPage from '../results.vue'
import { classifyResults, classifyNotes } from '../../state/results'
import type { ClassifyResponse } from '../../state/results'

const { mockReplace, mockPush } = vi.hoisted(() => ({
  mockReplace: vi.fn(),
  mockPush: vi.fn(),
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ replace: mockReplace, push: mockPush }),
}))

const validResult: ClassifyResponse = {
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
  mockReplace.mockReset()
  mockPush.mockReset()
  classifyNotes.value = ''
})

describe('results page', () => {
  it('redirects to / when classifyResults is null', () => {
    classifyResults.value = null
    mount(ResultsPage)
    expect(mockReplace).toHaveBeenCalledWith('/')
  })

  it('does not redirect when classifyResults is set', () => {
    classifyResults.value = validResult
    mount(ResultsPage)
    expect(mockReplace).not.toHaveBeenCalled()
  })

  it('renders notes when classifyNotes is non-empty', () => {
    classifyResults.value = validResult
    classifyNotes.value = 'test notes'
    const wrapper = mount(ResultsPage)
    expect(wrapper.html()).toContain('Your notes: test notes')
  })

  it('renders a success card with correct content', () => {
    classifyResults.value = validResult
    const wrapper = mount(ResultsPage)
    expect(wrapper.html()).toContain('Level 1 — Inspection recommended')
    expect(wrapper.html()).toContain('92.5% confidence')
    expect(wrapper.html()).toContain('crack.jpg')
    expect(wrapper.html()).toContain('Minor crack')
    expect(wrapper.html()).toContain('Schedule inspection')
  })

  it('renders an error card when ok is false', () => {
    classifyResults.value = {
      results: [
        {
          filename: 'bad.jpg',
          ok: false,
          result: null,
          error: 'Decode failed',
        },
      ],
    }
    const wrapper = mount(ResultsPage)
    expect(wrapper.html()).toContain('Could not process image')
    expect(wrapper.html()).toContain('bad.jpg')
    expect(wrapper.html()).toContain('Decode failed')
  })
})
