<template>
  <main class="page">
    <a class="back-link" href="#" @click.prevent="goBack">← Back</a>

    <p v-if="classifyNotes" class="notes-readback">Your notes: {{ classifyNotes }}</p>

    <div
      v-for="(item, index) in classifyResults?.results ?? []"
      :key="item.filename + index"
      class="card"
    >
      <template v-if="item.ok && item.result">
        <div
          class="card-header"
          :style="{
            background: colourMap[item.result.final_label].background,
            color: colourMap[item.result.final_label].color,
          }"
        >
          <span>{{ item.result.severity_level }} — {{ urgencyLabel[item.result.urgency] }}</span>
          <span>{{ item.result.confidence.toFixed(1) }}% confidence</span>
        </div>
        <div class="card-body">
          <p class="image-id">#{{ index + 1 }}</p>
          <p class="filename">{{ item.filename }}</p>
          <p class="summary">{{ item.result.customer_summary }}</p>
          <p class="action">{{ item.result.recommended_action }}</p>
          <p class="disclaimer">{{ item.result.disclaimer }}</p>
        </div>
      </template>

      <template v-else>
        <div
          class="card-header"
          :style="{
            background: colourMap.error.background,
            color: colourMap.error.color,
          }"
        >
          <span>Could not process image</span>
        </div>
        <div class="card-body">
          <p class="filename">{{ item.filename }}</p>
          <p class="error-message">{{ item.error }}</p>
        </div>
      </template>
    </div>
  </main>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { classifyResults, classifyNotes } from '../state/results'

const router = useRouter()

const colourMap: Record<string, { background: string; color: string }> = {
  level1:  { background: '#e8f5e9', color: '#2e7d32' },
  level2:  { background: '#fff8e1', color: '#f57f17' },
  level3:  { background: '#fff3e0', color: '#e65100' },
  unclear: { background: '#f5f5f5', color: '#616161' },
  error:   { background: '#ffebee', color: '#c62828' },
}

const urgencyLabel: Record<string, string> = {
  inspection_recommended: 'Inspection recommended',
  contact_soon:           'Contact a professional soon',
  contact_immediately:    'Contact a professional immediately',
  unable_to_assess:       'Unable to assess',
}

onMounted(() => {
  if (classifyResults.value === null) {
    router.replace('/')
  }
})

function goBack() {
  classifyResults.value = null
  router.push('/')
}
</script>

<style scoped>
.page {
  max-width: 700px;
  margin: 0 auto;
  padding: 2rem 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.back-link {
  color: inherit;
  text-decoration: none;
  align-self: flex-start;
}

.back-link:hover {
  text-decoration: underline;
}

.notes-readback {
  margin: 0;
  color: #555;
}

.card {
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
}

.card-header {
  display: flex;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  font-weight: 600;
}

.card-body {
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.card-body p {
  margin: 0;
}

.image-id {
  font-size: 1.25rem;
  font-weight: 700;
  color: #333;
}

.filename {
  font-size: 0.9rem;
  color: #555;
}

.summary,
.action {
  color: #333;
}

.disclaimer {
  font-size: 0.8rem;
  font-style: italic;
  color: #777;
}

.error-message {
  color: #c62828;
}
</style>
