import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import {
  ApiRequestError,
  createApiClient,
  decodeSseEvent,
  type EventSourceLike,
} from './api/client'
import type { ApiClient, ConfigResponse, SseEvent } from './api/types'

const config: ConfigResponse = { revision: 'r1', values: {}, schema: { fields: [] } }

function stubClient(overrides: Partial<ApiClient> = {}): ApiClient {
  return {
    get: vi.fn().mockResolvedValue(config),
    put: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
    upload: vi.fn(),
    subscribeJobEvents: vi.fn(() => () => undefined),
    ...overrides,
  }
}

describe('application shell', () => {
  it('navigates among all six views with accessible controls', async () => {
    const user = userEvent.setup()
    render(<App client={stubClient()} />)
    await screen.findByRole('heading', { name: 'Workflow' })

    for (const view of ['HukFlow Studio', 'Jobs', 'Settings', 'Voice Profiles', 'Dubbing Texts', 'Workflow']) {
      await user.click(screen.getByRole('button', { name: view }))
      expect(screen.getByRole('heading', { name: view })).toBeInTheDocument()
    }
  })

  it('shows shared loading and error states', async () => {
    const user = userEvent.setup()
    let reject!: (reason: unknown) => void
    const pending = new Promise<ConfigResponse>((_, rejectPromise) => {
      reject = rejectPromise
    })
    render(<App client={stubClient({ get: vi.fn(() => pending) as ApiClient['get'] })} />)
    await user.click(screen.getByRole('button', { name: 'Workflow' }))
    expect(screen.getByRole('status')).toHaveTextContent('Loading application')

    reject(new Error('offline'))
    expect(await screen.findByRole('alert')).toHaveTextContent('offline')
  })

  it('switches to JobProgressView when a dubbing job is started and transitions to Studio upon completion', async () => {
    const user = userEvent.setup()
    let sseCallback: ((event: SseEvent) => void) | undefined
    const workflowConfig: ConfigResponse = {
      revision: 'r1',
      values: { source_language: 'en', target_language: 'be' },
      schema: {
        fields: [
          { name: 'source_language', label: 'Source language', type: 'select', workflow: true },
          { name: 'target_language', label: 'Target language', type: 'select', workflow: true },
        ],
      },
    }

    const client = stubClient({
      get: vi.fn(async (path: string) => {
        if (path === '/api/config') return workflowConfig
        if (path === '/api/options') return { languages: [{ value: 'en', label: 'English' }, { value: 'be', label: 'Belarusian' }] }
        if (path === '/api/jobs/job-app-1') return { id: 'job-app-1', status: 'running' }
        if (path === '/api/projects') return { projects: [] }
        return {}
      }) as ApiClient['get'],
      post: vi.fn().mockResolvedValue({ id: 'job-app-1', status: 'queued' }),
      upload: vi.fn().mockResolvedValue({ id: 'upload-video-1' }),
      subscribeJobEvents: vi.fn((_id, onEvent) => {
        sseCallback = onEvent
        return () => { sseCallback = undefined }
      }),
    })

    render(<App client={client} />)
    expect(await screen.findByRole('heading', { name: 'Workflow' })).toBeInTheDocument()

    // Upload video and start dubbing
    await user.upload(screen.getByLabelText('Video'), new File(['test'], 'input.mp4', { type: 'video/mp4' }))
    await user.click(screen.getByRole('button', { name: 'Start' }))

    // Expect transition to Progress Screen
    expect(await screen.findByRole('progressbar')).toBeInTheDocument()
    expect(screen.getByText('AI DUBBING PIPELINE')).toBeInTheDocument()
    expect(screen.getByText(/ID: job-app-1/i)).toBeInTheDocument()
    expect(screen.getByText('CURRENT ACTION:')).toBeInTheDocument()

    // Emit success event and click proceed to studio
    act(() => {
      sseCallback?.({
        id: 10,
        job_id: 'job-app-1',
        type: 'state',
        timestamp: '2026-08-16T20:00:10Z',
        data: { status: 'succeeded' },
      })
    })

    expect(await screen.findByText('Dubbing successfully generated!')).toBeInTheDocument()
    const proceedBtn = screen.getByRole('button', { name: /Go to Studio/i })
    await user.click(proceedBtn)

    // Should be on HukFlow Studio view now
    expect(await screen.findByRole('heading', { name: 'HukFlow Studio' })).toBeInTheDocument()
  })
})

describe('API client', () => {
  afterEach(() => vi.useRealTimers())

  it('decodes the documented SSE DTO', () => {
    expect(decodeSseEvent(JSON.stringify({
      id: 2,
      job_id: 'job-1',
      type: 'state',
      timestamp: '2026-08-13T10:00:00Z',
      data: { percent: 50 },
    }))).toEqual({
      id: 2,
      job_id: 'job-1',
      type: 'state',
      timestamp: '2026-08-13T10:00:00Z',
      data: { percent: 50 },
    })
    expect(() => decodeSseEvent('{"type":"progress"}')).toThrow('Invalid SSE event')
  })

  it('normalizes structured and unstructured API failures', async () => {
    const structuredFetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      code: 'invalid_language', message: 'Unsupported language', field: 'target_language', details: { value: 'xx' },
    }), { status: 422, headers: { 'content-type': 'application/json' } }))
    const structured = createApiClient({ fetch: structuredFetch })
    await expect(structured.get('/api/config')).rejects.toMatchObject({
      name: 'ApiRequestError', code: 'invalid_language', message: 'Unsupported language',
      field: 'target_language', details: { value: 'xx' }, status: 422,
    } satisfies Partial<ApiRequestError>)

    const plain = createApiClient({
      fetch: vi.fn().mockResolvedValue(new Response('Gateway unavailable', { status: 502 })),
    })
    await expect(plain.get('/api/config')).rejects.toMatchObject({
      code: 'http_502', message: 'Gateway unavailable', status: 502,
    })
  })

  it('reconnects SSE with the last received event id', async () => {
    vi.useFakeTimers()
    const sources: FakeEventSource[] = []
    const client = createApiClient({
      fetch: vi.fn(),
      eventSourceFactory: (url) => {
        const source = new FakeEventSource(url)
        sources.push(source)
        return source
      },
      reconnectDelayMs: 10,
    })
    const received = vi.fn()

    const close = client.subscribeJobEvents('job 1', received)
    sources[0].namedMessage('log', { id: 9, job_id: 'job 1', type: 'log', timestamp: 'now', data: {} })
    sources[0].fail()
    await act(() => vi.advanceTimersByTimeAsync(10))

    expect(received).toHaveBeenCalledWith(expect.objectContaining({ id: 9 }))
    expect(sources[1].url).toBe('/api/jobs/job%201/events?last_event_id=9')
    close()
    expect(sources[1].closed).toBe(true)
  })
})

class FakeEventSource implements EventSourceLike {
  onmessage: ((event: MessageEvent<string>) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  closed = false
  private readonly listeners = new Map<string, Array<(event: MessageEvent<string>) => void>>()

  constructor(readonly url: string) {}

  message(value: object) {
    this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent<string>)
  }

  addEventListener(type: string, listener: (event: MessageEvent<string>) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener])
  }

  namedMessage(type: string, value: object) {
    const event = { data: JSON.stringify(value) } as MessageEvent<string>
    for (const listener of this.listeners.get(type) ?? []) listener(event)
  }

  fail() {
    this.onerror?.(new Event('error'))
  }

  close() {
    this.closed = true
  }
}
