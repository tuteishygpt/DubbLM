export type JsonPrimitive = string | number | boolean | null
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue }

export interface ApiErrorBody {
  code: string
  message: string
  field?: string
  details?: JsonValue
}

export interface SelectOption {
  value: string
  label: string
  depends_on?: Record<string, string>
}

export interface SchemaFieldDefinition {
  name: string
  label?: string
  type: string
  required?: boolean
  description?: string
  section?: string
  scope?: string
  workflow?: boolean
  read_only?: boolean
  minimum?: number
  maximum?: number
  options?: Array<SelectOption | string>
  options_key?: string
}

export interface ApiSchema {
  fields: SchemaFieldDefinition[]
}

export interface ConfigResponse {
  revision: string
  values: Record<string, JsonValue>
  schema: ApiSchema
}

export interface OptionsResponse {
  [key: string]: JsonValue
}

export interface SseEvent {
  id: number
  job_id: string
  type: string
  timestamp: string
  data: Record<string, JsonValue>
}

export type ConnectionState = 'connected' | 'reconnecting'

export interface ApiClient {
  get<T = unknown>(path: string): Promise<T>
  put<T = unknown>(path: string, body?: unknown): Promise<T>
  post<T = unknown>(path: string, body?: unknown): Promise<T>
  delete<T = unknown>(path: string, body?: unknown): Promise<T>
  upload<T = unknown>(path: string, data: FormData): Promise<T>
  subscribeJobEvents(
    jobId: string,
    onEvent: (event: SseEvent) => void,
    options?: { onState?: (state: ConnectionState) => void; onError?: (error: Error) => void },
  ): () => void
}
