import { useEffect, useRef, useState } from 'react'
import { useClassifier } from '../lib/useClassifier'
import styles from './UploadPage.module.css'

const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp'])

const cx = (...classes: Array<string | false | undefined>) => classes.filter(Boolean).join(' ')

export function UploadPage() {
  const { loading, error, classify, clearError } = useClassifier()

  const fileInputRef = useRef<HTMLInputElement>(null)
  const dragDepth = useRef(0)
  const previewsRef = useRef<string[]>([])

  const [isDragging, setIsDragging] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  const [previews, setPreviews] = useState<string[]>([])
  const [notes, setNotes] = useState('')

  previewsRef.current = previews

  useEffect(() => () => previewsRef.current.forEach((url) => URL.revokeObjectURL(url)), [])

  function addFiles(incoming: FileList | File[]) {
    clearError()
    const accepted = Array.from(incoming).filter((file) => ALLOWED_TYPES.has(file.type))
    if (accepted.length === 0) return
    setFiles((prev) => [...prev, ...accepted])
    setPreviews((prev) => [...prev, ...accepted.map((file) => URL.createObjectURL(file))])
  }

  function removeFile(index: number) {
    clearError()
    URL.revokeObjectURL(previews[index])
    setFiles((prev) => prev.filter((_, i) => i !== index))
    setPreviews((prev) => prev.filter((_, i) => i !== index))
  }

  function onDragEnter(e: React.DragEvent) {
    e.preventDefault()
    dragDepth.current++
    setIsDragging(true)
  }

  function onDragLeave(e: React.DragEvent) {
    e.preventDefault()
    if (--dragDepth.current === 0) setIsDragging(false)
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    dragDepth.current = 0
    setIsDragging(false)
    if (loading) return
    if (e.dataTransfer?.files) addFiles(e.dataTransfer.files)
  }

  function openPicker() {
    if (!loading) fileInputRef.current?.click()
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      openPicker()
    }
  }

  function onFileInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files) addFiles(e.target.files)
    e.target.value = ''
  }

  async function submit() {
    await classify(files, notes)
  }

  return (
    <main className={styles.page}>
      <div className={styles.title}>
        <h1 className={styles.heading}>Foundation Crack Classifier</h1>
        <p className={styles.subtitle}>
          Upload photos of your foundation cracks for an AI-powered severity assessment.
        </p>
      </div>

      <div
        role="button"
        tabIndex={0}
        className={cx(
          styles.dropZone,
          isDragging && styles.dropZoneActive,
          loading && styles.dropZoneLoading,
        )}
        onDragOver={(e) => e.preventDefault()}
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={openPicker}
        onKeyDown={onKeyDown}
      >
        <p>
          <span className={styles.droptext}>Drag and drop images here</span> or{' '}
          <span className={styles.link}>click to browse</span>
        </p>
        <p className={styles.dropHint}>JPEG, PNG, or WebP</p>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        multiple
        disabled={loading}
        style={{ display: 'none' }}
        onChange={onFileInputChange}
      />

      {files.length > 0 && (
        <div className={styles.thumbnails}>
          {files.map((file, index) => (
            <div key={file.name + index} className={styles.thumbnail}>
              <img src={previews[index]} alt={`Image ${index + 1}`} />
              <span className={styles.thumbnailBadge}>#{index + 1}</span>
              <button
                className={styles.thumbnailRemove}
                disabled={loading}
                aria-label={`Remove image ${index + 1}`}
                onClick={(e) => {
                  e.stopPropagation()
                  removeFile(index)
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <div className={styles.notes}>
        <label htmlFor="notes">
          Notes <span>(optional)</span>
        </label>
        <p className={styles.notesDescription}>
          Add notes to help add context to the photos — physical description, where it is, how large
          the crack is, etc.
        </p>
        <textarea
          id="notes"
          className={styles.notesInput}
          value={notes}
          disabled={loading}
          placeholder="e.g. image #1 shows a 2 foot crack, image #2 shows a partially collapsed wall."
          onChange={(e) => setNotes(e.target.value)}
        />
      </div>

      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}

      <button className={styles.submit} disabled={files.length === 0 || loading} onClick={submit}>
        {loading ? 'Classifying…' : 'Classify'}
      </button>
    </main>
  )
}
