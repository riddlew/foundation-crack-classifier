import { afterEach, describe, expect, it, vi } from 'vitest'
import { getUrls, setBatch } from '../imageStore'

function file(name: string) {
  return new File(['bytes'], name, { type: 'image/jpeg' })
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('imageStore', () => {
  it('returns one object URL per file for a stored batch', () => {
    const id = setBatch([file('a.jpg'), file('b.jpg')])
    const urls = getUrls(id)
    expect(urls).toHaveLength(2)
    expect(urls?.every((u) => typeof u === 'string')).toBe(true)
  })

  it('returns null for an unknown batch id', () => {
    setBatch([file('a.jpg')])
    expect(getUrls('not-a-real-id')).toBeNull()
  })

  it('revokes the previous batch object URLs when a new batch is set', () => {
    const revoke = vi.spyOn(URL, 'revokeObjectURL')
    const firstId = setBatch([file('a.jpg'), file('b.jpg')])
    const firstUrls = getUrls(firstId)!
    revoke.mockClear()

    setBatch([file('c.jpg')])

    expect(revoke).toHaveBeenCalledTimes(2)
    for (const url of firstUrls) {
      expect(revoke).toHaveBeenCalledWith(url)
    }
  })

  it('makes the previous batch id unresolvable after replacement', () => {
    const firstId = setBatch([file('a.jpg')])
    setBatch([file('b.jpg')])
    expect(getUrls(firstId)).toBeNull()
  })
})
