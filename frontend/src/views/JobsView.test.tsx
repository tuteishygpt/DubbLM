import { act, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { ApiClient, ConnectionState, SseEvent } from '../api/types'
import { JobsView } from './JobsView'

describe('JobsView', () => {
  it('renders registered files and applies named state, file, log, error, and snapshot events', async () => {
    const emitters = new Map<string, (event: SseEvent) => void>()
    let connection!: (state: ConnectionState) => void
    const client: ApiClient = {
      get: vi.fn().mockResolvedValue({ jobs: [
        { id: 'job-q', status: 'queued', files: [] },
        { id: 'job-r', status: 'running', files: [] },
        { id: 'job-c', status: 'succeeded', files: [
          { id: 'result-1', name: 'dubbed.mp4', kind: 'result', size: 10 },
          { id: 'report-1', name: 'report.txt', kind: 'report', size: 5 },
        ] },
        { id: 'job-f', status: 'failed', files: [] },
      ] }),
      put: vi.fn(), post: vi.fn(), delete: vi.fn(), upload: vi.fn(),
      subscribeJobEvents: vi.fn((id, onEvent, options) => {
        emitters.set(id, onEvent)
        connection = options!.onState!
        return () => undefined
      }),
    }
    render(<JobsView client={client} />)

    for (const status of ['queued', 'running', 'succeeded', 'failed']) expect(await screen.findByText(status, { exact: true })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'dubbed.mp4' })).toHaveAttribute('href', '/api/jobs/job-c/files/result-1')
    expect(screen.getByRole('link', { name: 'report.txt' })).toHaveAttribute('href', '/api/jobs/job-c/files/report-1')

    act(() => connection('reconnecting'))
    expect(screen.getByRole('status')).toHaveTextContent('Reconnecting')
    act(() => emitters.get('job-r')!({ id: 1, job_id: 'job-r', type: 'log', timestamp: 'now', data: { message: 'Separating speakers' } }))
    expect(screen.getByText('Separating speakers')).toBeInTheDocument()
    act(() => emitters.get('job-r')!({ id: 2, job_id: 'job-r', type: 'file', timestamp: 'now', data: { id: 'artifact-1', name: 'captions.srt', kind: 'artifact', size: 3 } }))
    expect(screen.getByRole('link', { name: 'captions.srt' })).toHaveAttribute('href', '/api/jobs/job-r/files/artifact-1')
    act(() => emitters.get('job-r')!({ id: 3, job_id: 'job-r', type: 'state', timestamp: 'later', data: { status: 'succeeded' } }))
    expect(screen.getByText('succeeded', { selector: '[data-job-id="job-r"] *' })).toBeInTheDocument()

    act(() => emitters.get('job-q')!({ id: 4, job_id: 'job-q', type: 'snapshot', timestamp: 'later', data: {
      status: 'failed', files: [], log_tail: [{ id: 3, message: 'Recovered log' }], error: { code: 'pipeline_failed', message: 'Bad input' },
    } }))
    expect(screen.getByText('failed', { selector: '[data-job-id="job-q"] *' })).toBeInTheDocument()
    expect(screen.getByText('Recovered log')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('Bad input')
  })
})
