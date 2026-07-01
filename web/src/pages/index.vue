<template>
  <main class="page">
    <h1>Foundation Crack Classifier</h1>
    <p class="subtitle">Upload photos of foundation cracks for a severity assessment.</p>

    <div
      role="button"
      tabindex="0"
      class="drop-zone"
      :class="{ 'drop-zone--active': isDragging, 'drop-zone--loading': loading }"
      @dragover.prevent
      @dragenter.prevent="onDragEnter"
      @dragleave.prevent="onDragLeave"
      @drop.prevent="onDrop"
      @click="!loading && fileInput?.click()"
      @keydown.enter.space.prevent="!loading && fileInput?.click()"
    >
      <p>Drop images here or <span class="link">click to browse</span></p>
      <p class="drop-hint">JPEG, PNG, or WebP</p>
    </div>

    <input
      ref="fileInput"
      type="file"
      accept="image/jpeg,image/png,image/webp"
      multiple
      :disabled="loading"
      style="display: none"
      @change="onFileInputChange"
    />

    <div v-if="files.length > 0" class="thumbnails">
      <div v-for="(file, index) in files" :key="file.name + index" class="thumbnail">
        <img :src="previews[index]" :alt="`Image ${index + 1}`" />
        <span class="thumbnail-badge">#{{ index + 1 }}</span>
        <button class="thumbnail-remove" :disabled="loading" :aria-label="`Remove image ${index + 1}`" @click.stop="removeFile(index)">×</button>
      </div>
    </div>

    <label for="notes" class="sr-only">Notes</label>
    <textarea
      id="notes"
      v-model="notes"
      class="notes"
      :disabled="loading"
      placeholder="Optional — e.g. image #1 shows the east wall, image #2 shows the crack near the window."
    ></textarea>

    <p v-if="error" class="error" role="alert">{{ error }}</p>

    <button class="submit" :disabled="files.length === 0 || loading" @click="submit">
      {{ loading ? 'Classifying…' : 'Classify' }}
    </button>
  </main>
</template>

<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { useClassifier } from '../composables/useClassifier'

const { loading, error, classify } = useClassifier()

const fileInput = ref<HTMLInputElement | null>(null)
const isDragging = ref(false)
const dragDepth = ref(0)
const files = ref<File[]>([])
const previews = ref<string[]>([])
const notes = ref('')

const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp'])

function addFiles(incoming: FileList | File[]) {
  error.value = null  // clear stale error
  for (const file of Array.from(incoming)) {
    if (!ALLOWED_TYPES.has(file.type)) continue
    files.value.push(file)
    previews.value.push(URL.createObjectURL(file))
  }
}

function removeFile(index: number) {
  error.value = null  // clear stale error
  URL.revokeObjectURL(previews.value[index])
  files.value.splice(index, 1)
  previews.value.splice(index, 1)
}

function onDragEnter() {
  dragDepth.value++
  isDragging.value = true
}

function onDragLeave() {
  if (--dragDepth.value === 0) isDragging.value = false
}

function onDrop(e: DragEvent) {
  dragDepth.value = 0
  isDragging.value = false
  if (loading.value) return
  if (e.dataTransfer?.files) addFiles(e.dataTransfer.files)
}

function onFileInputChange(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files) addFiles(input.files)
  input.value = ''
}

async function submit() {
  await classify(files.value, notes.value)
}

onUnmounted(() => {
  for (const url of previews.value) URL.revokeObjectURL(url)
})
</script>

<style scoped>
.page {
  max-width: 600px;
  margin: 0 auto;
  padding: 2rem 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

h1 { margin: 0; }

.subtitle {
  margin: 0;
  color: #666;
}

.drop-zone {
  border: 2px dashed #ccc;
  border-radius: 8px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
  user-select: none;
}

.drop-zone--active {
  border-color: #555;
  background: #f9f9f9;
}

.drop-zone--loading {
  cursor: not-allowed;
  opacity: 0.6;
}

.drop-zone:focus-visible {
  outline: 2px solid #333;
  outline-offset: 2px;
}

.drop-zone p { margin: 0; }

.link { text-decoration: underline; }

.drop-zone .drop-hint {
  margin-top: 0.25rem;
  font-size: 0.85rem;
  color: #999;
}

.thumbnails {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.thumbnail {
  position: relative;
  width: 80px;
  height: 80px;
}

.thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 4px;
  display: block;
}

.thumbnail-badge {
  position: absolute;
  bottom: 4px;
  left: 4px;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  font-size: 0.7rem;
  padding: 1px 4px;
  border-radius: 3px;
  pointer-events: none;
}

.thumbnail-remove {
  position: absolute;
  top: 3px;
  right: 3px;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  border: none;
  border-radius: 3px;
  width: 18px;
  height: 18px;
  font-size: 0.85rem;
  line-height: 1;
  cursor: pointer;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.thumbnail-remove:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.notes {
  width: 100%;
  min-height: 80px;
  padding: 0.5rem;
  border: 1px solid #ccc;
  border-radius: 6px;
  font-family: inherit;
  font-size: 0.9rem;
  resize: vertical;
  box-sizing: border-box;
}

.notes:disabled {
  background: #f5f5f5;
  color: #999;
}

.error {
  margin: 0;
  color: #c62828;
}

.submit {
  align-self: flex-end;
  padding: 0.5rem 1.5rem;
  background: #333;
  color: #fff;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  cursor: pointer;
}

.submit:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.submit:not(:disabled):hover {
  background: #111;
}
</style>
