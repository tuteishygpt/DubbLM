import type { ApiClient, ApiErrorBody, JsonValue, SseEvent } from './types'

export interface EventSourceLike {
  onmessage: ((event: MessageEvent<string>) => void) | null
  onerror: ((event: Event) => void) | null
  addEventListener(type: string, listener: (event: MessageEvent<string>) => void): void
  close(): void
}

interface ClientDependencies {
  fetch?: typeof fetch
  eventSourceFactory?: (url: string) => EventSourceLike
  reconnectDelayMs?: number
}

export class ApiRequestError extends Error {
  readonly name = 'ApiRequestError'

  constructor(
    message: string,
    readonly code: string,
    readonly status: number,
    readonly field?: string,
    readonly details?: JsonValue,
  ) {
    super(message)
  }
}

export function decodeSseEvent(input: string): SseEvent {
  let value: unknown
  try {
    value = JSON.parse(input)
  } catch {
    throw new Error('Invalid SSE event')
  }
  if (!isRecord(value)
    || typeof value.id !== 'number'
    || !Number.isSafeInteger(value.id)
    || value.id < 0
    || typeof value.job_id !== 'string'
    || typeof value.type !== 'string'
    || typeof value.timestamp !== 'string'
    || !isRecord(value.data)) {
    throw new Error('Invalid SSE event')
  }
  return value as unknown as SseEvent
}

export function createApiClient(dependencies: ClientDependencies = {}): ApiClient {
  const requestFetch = dependencies.fetch ?? globalThis.fetch.bind(globalThis)
  const eventSourceFactory: (url: string) => EventSourceLike = dependencies.eventSourceFactory
    ?? ((url: string) => new EventSource(url) as unknown as EventSourceLike)
  const reconnectDelayMs = dependencies.reconnectDelayMs ?? 1000

  async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const isForm = body instanceof FormData
    const response = await requestFetch(path, {
      method,
      headers: body === undefined || isForm ? undefined : { 'content-type': 'application/json' },
      body: body === undefined ? undefined : isForm ? body : JSON.stringify(body),
    })
    if (!response.ok) throw await normalizeFailure(response)
    if (response.status === 204) return undefined as T
    return await response.json() as T
  }

  return {
    get: <T>(path: string) => request<T>('GET', path),
    put: <T>(path: string, body?: unknown) => request<T>('PUT', path, body),
    post: <T>(path: string, body?: unknown) => request<T>('POST', path, body),
    delete: <T>(path: string, body?: unknown) => request<T>('DELETE', path, body),
    upload: <T>(path: string, data: FormData) => request<T>('POST', path, data),
    subscribeJobEvents(jobId, onEvent, options = {}) {
      let closed = false
      let source: EventSourceLike | undefined
      let reconnectTimer: ReturnType<typeof setTimeout> | undefined
      let lastEventId: number | undefined

      const connect = () => {
        const suffix = lastEventId !== undefined ? `?last_event_id=${lastEventId}` : ''
        const nextSource = eventSourceFactory(`/api/jobs/${encodeURIComponent(jobId)}/events${suffix}`)
        source = nextSource
        options.onState?.('connected')
        const receive = (message: MessageEvent<string>) => {
          try {
            const event = decodeSseEvent(message.data)
            lastEventId = event.id
            onEvent(event)
          } catch (error) {
            options.onError?.(error instanceof Error ? error : new Error(String(error)))
          }
        }
        for (const type of ['log', 'state', 'file', 'error', 'snapshot']) nextSource.addEventListener(type, receive)
        nextSource.onmessage = receive
        nextSource.onerror = (event) => {
          if ('data' in event) return
          nextSource.close()
          if (closed) return
          options.onState?.('reconnecting')
          reconnectTimer = setTimeout(connect, reconnectDelayMs)
        }
      }

      connect()
      return () => {
        closed = true
        if (reconnectTimer) clearTimeout(reconnectTimer)
        source?.close()
      }
    },
  }
}

async function normalizeFailure(response: Response): Promise<ApiRequestError> {
  const text = await response.text()
  let parsed: Partial<ApiErrorBody> = {}
  try {
    parsed = JSON.parse(text) as Partial<ApiErrorBody>
  } catch {
    // Plain-text responses still receive the same client error shape.
  }
  return new ApiRequestError(
    parsed.message ?? (text || response.statusText || `Request failed (${response.status})`),
    parsed.code ?? `http_${response.status}`,
    response.status,
    parsed.field,
    parsed.details,
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export const apiClient = createApiClient()
