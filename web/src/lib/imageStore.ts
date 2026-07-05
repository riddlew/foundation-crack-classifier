// Holds the object URLs for the most recently classified upload batch, keyed by
// a lightweight id. Only that id travels through router navigation state, so the
// image bytes never enter history.state. Bounded to a single batch: the previous
// batch's object URLs are revoked whenever a new batch is registered.

let counter = 0
let current: { id: string; urls: string[] } | null = null

export function setBatch(files: File[]): string {
  if (current) {
    current.urls.forEach((url) => URL.revokeObjectURL(url))
  }
  counter += 1
  const id = `batch-${counter}`
  current = { id, urls: files.map((file) => URL.createObjectURL(file)) }
  return id
}

export function getUrls(id: string): string[] | null {
  if (!current || current.id !== id) {
    return null
  }
  return current.urls
}
