import { FormEvent, useEffect, useState } from 'react'
import type { ApiClient, ConfigResponse, JsonValue, OptionsResponse, SchemaFieldDefinition, SelectOption } from '../api/types'

export function WorkflowView({ client, config }: { client: ApiClient; config: ConfigResponse }) {
  const fields = config.schema.fields.filter((field) => field.workflow || field.scope === 'workflow')
  const [options, setOptions] = useState<OptionsResponse>({})
  const [values, setValues] = useState<Record<string, JsonValue>>(() => Object.fromEntries(fields.map((field) => [field.name, config.values[field.name] ?? ''])))
  const [video, setVideo] = useState<File>()
  const [tracks, setTracks] = useState<File[]>([])
  const [mapping, setMapping] = useState('{}')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  useEffect(() => { client.get<OptionsResponse>('/api/options').then(setOptions).catch((reason) => setError(String(reason))) }, [client])

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')
    if (!video) { setError('Video is required'); return }
    let speakerLabelMapping: Record<string, string>
    try {
      const parsed: unknown = JSON.parse(mapping)
      if (!isStringRecord(parsed)) throw new Error()
      speakerLabelMapping = parsed
    } catch { setError('Speaker-label mapping must be a JSON object of speaker IDs to file names'); return }
    try {
      const videoUpload = await upload(client, video)
      const uploadedTracks = new Map<string, string>()
      for (const track of tracks) uploadedTracks.set(track.name, (await upload(client, track)).id)
      const isolatedTracks = Object.fromEntries(Object.entries(speakerLabelMapping).map(([speaker, fileName]) => {
        const uploadId = uploadedTracks.get(fileName)
        if (!uploadId) throw new Error(`No isolated track named ${fileName}`)
        return [speaker, uploadId]
      }))
      const overrides = Object.fromEntries(fields
        .filter((field) => values[field.name] !== config.values[field.name])
        .map((field) => [field.name, values[field.name]]))
      const job = await client.post<{ id: string; status: string }>('/api/jobs', {
        input_upload_id: videoUpload.id,
        isolated_tracks: isolatedTracks,
        overrides,
      })
      setMessage(`Job ${job.id} ${job.status}`)
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }

  return <section>
    <h1>Workflow</h1>
    {error && <p role="alert">{error}</p>}
    {message && <p role="status">{message}</p>}
    <form onSubmit={submit}>
      <label>Video<input type="file" accept="video/*" onChange={(e) => setVideo(e.target.files?.[0])} /></label>
      <label>Isolated audio track<input type="file" accept="audio/*" multiple onChange={(e) => setTracks(Array.from(e.target.files ?? []))} /></label>
      <label>Speaker-label mapping<textarea value={mapping} onChange={(e) => setMapping(e.target.value)} /></label>
      {fields.map((field) => <WorkflowField key={field.name} field={field} value={values[field.name]} options={options} onChange={(value) => setValues((current) => ({ ...current, [field.name]: value }))} />)}
      <button type="submit">Queue job</button>
    </form>
  </section>
}

function WorkflowField({ field, value, options, onChange }: { field: SchemaFieldDefinition; value: JsonValue; options: OptionsResponse; onChange(value: JsonValue): void }) {
  const label = field.label ?? field.name
  if (field.type === 'boolean') return <label><input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />{label}</label>
  if (field.type === 'select') {
    const choices = normalizeOptions(field.options ?? options[field.options_key ?? field.name])
    return <label>{label}<select value={String(value ?? '')} onChange={(e) => onChange(e.target.value)}>{choices.map((choice) => <option key={choice.value} value={choice.value}>{choice.label}</option>)}</select></label>
  }
  return <label>{label}<input type={field.type === 'number' ? 'number' : 'text'} value={String(value ?? '')} onChange={(e) => onChange(field.type === 'number' ? Number(e.target.value) : e.target.value)} /></label>
}

function normalizeOptions(value: JsonValue | Array<SelectOption | string> | undefined): SelectOption[] {
  if (!Array.isArray(value)) return []
  return value.map((item) => typeof item === 'string' ? { value: item, label: item } : item as unknown as SelectOption)
}

async function upload(client: ApiClient, file: File) {
  const form = new FormData(); form.append('file', file)
  return client.upload<{ id: string }>('/api/uploads', form)
}

function isStringRecord(value: unknown): value is Record<string, string> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    && Object.entries(value).every(([speaker, fileName]) => speaker.length > 0 && typeof fileName === 'string' && fileName.length > 0)
}
