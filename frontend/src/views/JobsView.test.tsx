import { act, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { ApiClient, ConnectionState, SseEvent } from '../api/types'
import { JobsView } from './JobsView'

describe('JobsView', () => {
  it('shows queued, running, and terminal jobs with live reconnecting logs and artifacts', async () => {
    let emit!: (event: SseEvent) => void
    let connection!: (state: ConnectionState) => void
    const client: ApiClient = {
      get: vi.fn().mockResolvedValue({ jobs: [
        { id: 'job-q', status: 'queued' },
        { id: 'job-r', status: 'running' },
        { id: 'job-c', status: 'completed', result_url: '/results/job-c', report_url: '/reports/job-c', artifacts: [{ name: 'dubbed.mp4', url: '/artifacts/dubbed.mp4' }] },
        { id: 'job-f', status: 'failed' },
        { id: 'job-x', status: 'canceled' },
      ] }),
      put: vi.fn(), post: vi.fn(), delete: vi.fn(), upload: vi.fn(),
      subscribeJobEvents: vi.fn((_id, onEvent, options) => {
        emit = onEvent
        connection = options!.onState!
        return () => undefined
      }),
    }
    render(<JobsView client={client} />)

    for (const status of ['queued', 'running', 'completed', 'failed', 'canceled']) {
      expect(await screen.findByText(status, { exact: true })).toBeInTheDocument()
    }
    expect(screen.getByRole('link', { name: 'Result' })).toHaveAttribute('href', '/results/job-c')
    expect(screen.getByRole('link', { name: 'Report' })).toHaveAttribute('href', '/reports/job-c')
    expect(screen.getByRole('link', { name: 'dubbed.mp4' })).toHaveAttribute('href', '/artifacts/dubbed.mp4')

    act(() => connection('reconnecting'))
    expect(screen.getByRole('status')).toHaveTextContent('Reconnecting')
    act(() => emit({ id: 'e1', job_id: 'job-r', type: 'log', timestamp: 'now', data: { message: 'Separating speakers' } }))
    expect(screen.getByText('Separating speakers')).toBeInTheDocument()
    act(() => emit({ id: 'e2', job_id: 'job-r', type: 'status', timestamp: 'later', data: { status: 'completed' } }))
    expect(screen.getByText('completed', { selector: '[data-job-id="job-r"] *' })).toBeInTheDocument()
  })
})
