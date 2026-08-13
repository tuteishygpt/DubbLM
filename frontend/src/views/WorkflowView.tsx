import { FormEvent, useEffect, useState } from 'react'
import type { ApiClient, ConfigResponse, JsonValue, OptionsResponse, SchemaFieldDefinition, SelectOption } from '../api/types'

export function WorkflowView({ client, config }: { client: ApiClient; config: ConfigResponse }) {
  const fields = config.schema.fields.filter((field) => field.workflow || field.scope === 'workflow')
  const [options, setOptions] = useState<OptionsResponse>({})
  const [values, setValues] = useState<Record<string, JsonValue>>(() => Object.fromEntries(fields.map((field) => [field.name, config.values[field.name] ?? ''])))
  const [video, setVideo] = useState<File>()
  const [track, setTrack] = useState<File>()
  const [mapping, setMapping] = useState('{}')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  useEffect(() => { client.get<OptionsResponse>('/api/options').then(setOptions).catch((reason) => setError(String(reason))) }, [client])

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')
    if (!video) { setError('Video is required'); return }
    let speakerLabelMapping: unknown
    try { speakerLabelMapping = JSON.parse(mapping) } catch { setError('Speaker-label mapping must be valid JSON'); return }
    try {
      const videoUpload = await upload(client, '/api/uploads/video', video)
      const trackUpload = track ? await upload(client, '/api/uploads/isolated-track', track) : undefined
      const overrides = Object.fromEntries(fields
        .filter((field) => values[field.name] !== config.values[field.name])
        .map((field) => [field.name, values[field.name]]))
      const job = await client.post<{ id: string; status: string }>('/api/jobs', {
        video_upload_id: videoUpload.id,
        ...(trackUpload ? { isolated_track_upload_id: trackUpload.id } : {}),
        speaker_label_mapping: speakerLabelMapping,
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
      <label>Isolated audio track<input type="file" accept="audio/*" onChange={(e) => setTrack(e.target.files?.[0])} /></label>
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

async function upload(client: ApiClient, path: string, file: File) {
  const form = new FormData(); form.append('file', file)
  return client.upload<{ id: string }>(path, form)
}
