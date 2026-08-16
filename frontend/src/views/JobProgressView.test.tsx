import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ApiClient, SseEvent } from '../api/types'
import { JobProgressView, PIPELINE_STAGES } from './JobProgressView'

function createTestClient(overrides: Partial<ApiClient> = {}) {
  let eventCallback: ((event: SseEvent) => void) | undefined
  const client: ApiClient = {
    get: vi.fn().mockResolvedValue({ id: 'job-test-1', status: 'running' }),
    put: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
    upload: vi.fn(),
    subscribeJobEvents: vi.fn((_jobId, onEvent) => {
      eventCallback = onEvent
      return () => {
        eventCallback = undefined
      }
    }),
    ...overrides,
  }

  const emitEvent = (event: SseEvent) => {
    act(() => {
      eventCallback?.(event)
    })
  }

  return { client, emitEvent }
}

describe('JobProgressView', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders all 6 pipeline stages with descriptions, project title, and initial progress bar', async () => {
    const { client } = createTestClient()
    const onComplete = vi.fn()
    const onCancel = vi.fn()

    render(
      <JobProgressView
        client={client}
        jobId="job-123"
        projectName="Test Belarusian Dubbing"
        onComplete={onComplete}
        onCancel={onCancel}
      />,
    )

    expect(screen.getByText('Test Belarusian Dubbing')).toBeInTheDocument()
    expect(screen.getByText(/ID: job-123/i)).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
    expect(screen.getByText('CURRENT ACTION:')).toBeInTheDocument()

    // Verify all 6 stages are listed
    for (const stage of PIPELINE_STAGES) {
      expect(screen.getAllByText(stage.title).length).toBeGreaterThan(0)
      expect(screen.getByText(stage.description)).toBeInTheDocument()
    }
  })

  it('updates stage progress and active action description dynamically as log events arrive', async () => {
    const { client, emitEvent } = createTestClient()
    render(
      <JobProgressView
        client={client}
        jobId="job-123"
        projectName="Podcast Dubbing"
        onComplete={vi.fn()}
      />,
    )

    // Stage 1: Audio extraction log
    emitEvent({
      id: 1,
      job_id: 'job-123',
      type: 'log',
      timestamp: '2026-08-16T20:00:00Z',
      data: { message: 'Separating audio and isolating background noise...' },
    })

    await waitFor(() => {
      expect(screen.getAllByText(/Separating audio and isolating background noise/i).length).toBeGreaterThan(0)
    })

    // Stage 2: Diarization & Transcription
    emitEvent({
      id: 2,
      job_id: 'job-123',
      type: 'log',
      timestamp: '2026-08-16T20:00:05Z',
      data: { message: 'Diarization complete, transcribing audio with WhisperX...' },
    })

    await waitFor(() => {
      expect(screen.getAllByText(/Diarization complete, transcribing/i).length).toBeGreaterThan(0)
    })

    // Stage 3: Translation
    emitEvent({
      id: 3,
      job_id: 'job-123',
      type: 'log',
      timestamp: '2026-08-16T20:00:10Z',
      data: { message: 'Translating segments into Belarusian with LLM...' },
    })

    await waitFor(() => {
      expect(screen.getAllByText(/Translating segments into Belarusian/i).length).toBeGreaterThan(0)
    })

    // Stage 5: Speech synthesis
    emitEvent({
      id: 4,
      job_id: 'job-123',
      type: 'log',
      timestamp: '2026-08-16T20:00:15Z',
      data: { message: 'Synthesizing speech chunks with Gemini TTS...' },
    })

    await waitFor(() => {
      expect(screen.getAllByText(/Synthesizing speech chunks with Gemini TTS/i).length).toBeGreaterThan(0)
    })
  })

  it('completes pipeline on state=succeeded, shows 100%, and transitions to Studio', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const { client, emitEvent } = createTestClient()
    const onComplete = vi.fn()

    render(
      <JobProgressView
        client={client}
        jobId="job-456"
        projectName="Commercial Ad"
        onComplete={onComplete}
      />,
    )

    // Job finishes successfully
    emitEvent({
      id: 10,
      job_id: 'job-456',
      type: 'state',
      timestamp: '2026-08-16T20:00:20Z',
      data: { status: 'succeeded' },
    })

    await waitFor(() => {
      expect(screen.getByText('Succeeded')).toBeInTheDocument()
      expect(screen.getByText('100%')).toBeInTheDocument()
      expect(screen.getByText('Dubbing successfully generated!')).toBeInTheDocument()
    })

    // Advance timer for auto redirect
    await act(async () => {
      vi.advanceTimersByTime(2500)
    })

    expect(onComplete).toHaveBeenCalledWith('job-456')
  })

  it('allows manual click to transition to Studio immediately upon completion', async () => {
    const user = userEvent.setup()
    const { client, emitEvent } = createTestClient()
    const onComplete = vi.fn()

    render(
      <JobProgressView
        client={client}
        jobId="job-789"
        projectName="Documentary"
        onComplete={onComplete}
      />,
    )

    emitEvent({
      id: 11,
      job_id: 'job-789',
      type: 'state',
      timestamp: '2026-08-16T20:00:25Z',
      data: { status: 'succeeded' },
    })

    await waitFor(() => {
      expect(screen.getByText('Dubbing successfully generated!')).toBeInTheDocument()
    })

    const studioBtn = screen.getByRole('button', { name: /Go to Studio/i })
    await user.click(studioBtn)

    expect(onComplete).toHaveBeenCalledWith('job-789')
  })

  it('displays error details when job fails and allows cancel/return', async () => {
    const user = userEvent.setup()
    const { client, emitEvent } = createTestClient()
    const onCancel = vi.fn()

    render(
      <JobProgressView
        client={client}
        jobId="job-err"
        projectName="Failed Run"
        onComplete={vi.fn()}
        onCancel={onCancel}
      />,
    )

    emitEvent({
      id: 12,
      job_id: 'job-err',
      type: 'error',
      timestamp: '2026-08-16T20:00:30Z',
      data: { message: 'Out of GPU memory during Demucs separation' },
    })

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Out of GPU memory during Demucs separation')
      expect(screen.getByText('Failed')).toBeInTheDocument()
    })

    const retryBtn = screen.getByRole('button', { name: /Return to Workflow and retry/i })
    await user.click(retryBtn)
    expect(onCancel).toHaveBeenCalled()
  })

  it('supports expanding and collapsing logs terminal', async () => {
    const user = userEvent.setup()
    const { client, emitEvent } = createTestClient()

    render(
      <JobProgressView
        client={client}
        jobId="job-logs"
        projectName="Log Test"
        onComplete={vi.fn()}
      />,
    )

    emitEvent({
      id: 1,
      job_id: 'job-logs',
      type: 'log',
      timestamp: '2026-08-16T20:00:00Z',
      data: { message: 'Pipeline initialized.' },
    })

    expect(screen.getByText('Pipeline initialized.')).toBeInTheDocument()

    const toggleBtn = screen.getByRole('button', { name: /Collapse \/ Expand Logs/i })
    await user.click(toggleBtn)
    expect(screen.getByText('Collapse')).toBeInTheDocument()
  })
})
