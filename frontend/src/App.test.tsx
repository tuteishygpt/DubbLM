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
import type { ApiClient, ConfigResponse } from './api/types'

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
  it('navigates among all five views with accessible controls', async () => {
    const user = userEvent.setup()
    render(<App client={stubClient()} />)
    await screen.findByRole('heading', { name: 'Workflow' })

    for (const view of ['Jobs', 'Settings', 'Voice Profiles', 'Dubbing Texts', 'Workflow']) {
      await user.click(screen.getByRole('button', { name: view }))
      expect(screen.getByRole('heading', { name: view })).toBeInTheDocument()
    }
  })

  it('shows shared loading and error states', async () => {
    let reject!: (reason: unknown) => void
    const pending = new Promise<ConfigResponse>((_, rejectPromise) => {
      reject = rejectPromise
    })
    render(<App client={stubClient({ get: vi.fn(() => pending) as ApiClient['get'] })} />)
    expect(screen.getByRole('status')).toHaveTextContent('Loading application')

    reject(new Error('offline'))
    expect(await screen.findByRole('alert')).toHaveTextContent('offline')
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
