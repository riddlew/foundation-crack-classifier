import { createFileRoute } from '@tanstack/react-router'
import { ResultsPage } from '../pages/ResultsPage'

export const Route = createFileRoute('/results')({
  component: ResultsPage,
})
