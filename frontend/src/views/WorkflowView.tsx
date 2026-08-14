import { FormEvent, useEffect, useState } from 'react'
import type { ApiClient, ConfigResponse, JsonValue, OptionsResponse, SchemaFieldDefinition, SelectOption } from '../api/types'

export function WorkflowView({ client, config }: { client: ApiClient; config: ConfigResponse }) {
  const fields = config.schema.fields.filter((field) => field.workflow || field.scope === 'workflow')
  const regularFields = fields.filter((f) => f.type !== 'boolean')
  const booleanFields = fields.filter((f) => f.type === 'boolean')

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

  return (
    <section>
      <div className="setup-wizard-card">
        {/* Wizard Header */}
        <div className="wizard-header">
          <div>
            <h1>Workflow</h1>
            <p>Configure source media and target languages for AI processing.</p>
          </div>
          <button className="header-icon-btn" type="button" title="Close">
            <span className="material-symbols-outlined" aria-hidden="true">close</span>
          </button>
        </div>

        {/* Wizard Stepper Progress Bar */}
        <div className="wizard-stepper">
          <div className="step-item active">
            <div className="step-badge">1</div>
            <span>Setup</span>
          </div>
          <div className="step-divider"></div>
          <div className="step-item">
            <div className="step-badge">2</div>
            <span>Extraction</span>
          </div>
          <div className="step-divider"></div>
          <div className="step-item">
            <div className="step-badge">3</div>
            <span>Diarization</span>
          </div>
          <div className="step-divider"></div>
          <div className="step-item">
            <div className="step-badge">4</div>
            <span>Translation</span>
          </div>
        </div>

        {/* Wizard Form Body */}
        <div className="wizard-body">
          {error && <p role="alert">{error}</p>}
          {message && <p role="status">{message}</p>}

          <form onSubmit={submit}>
            {/* Project Details Section */}
            <div className="form-section">
              <div className="form-section-title">
                <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>folder</span>
                Project Details
              </div>
              <div className="field-grid">
                <label>
                  Project Name
                  <input type="text" placeholder="e.g., Q3 Marketing Video" defaultValue="Project Alpha" />
                </label>
                <label>
                  Workspace
                  <select defaultValue="Personal Workspace">
                    <option value="Personal Workspace">Personal Workspace</option>
                    <option value="Team Alpha">Team Alpha</option>
                  </select>
                </label>
              </div>
            </div>

            {/* Media & Tracks Section */}
            <div className="form-section">
              <div className="form-section-title">
                <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>movie</span>
                Source Media & Isolated Tracks
              </div>

              <div className="stitch-dropzone">
                <div className="dropzone-icon">
                  <span className="material-symbols-outlined" style={{ fontSize: '28px' }}>upload_file</span>
                </div>
                <div className="dropzone-title">Drag and drop media files here</div>
                <div className="dropzone-desc">Supports MP4, MOV, WAV up to 5GB</div>
                <div className="field-grid" style={{ width: '100%' }}>
                  <label className="field-file">
                    Video
                    <input type="file" accept="video/*" onChange={(e) => setVideo(e.target.files?.[0])} />
                  </label>
                  <label className="field-file">
                    Isolated audio track
                    <input type="file" accept="audio/*" multiple onChange={(e) => setTracks(Array.from(e.target.files ?? []))} />
                  </label>
                </div>
              </div>

              <label className="field-code">
                Speaker-label mapping
                <textarea value={mapping} onChange={(e) => setMapping(e.target.value)} />
              </label>
            </div>

            {/* Job Parameters / Options */}
            {regularFields.length > 0 && (
              <div className="form-section">
                <div className="form-section-title">
                  <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>tune</span>
                  Language & Pipeline Parameters
                </div>
                <div className="field-grid">
                  {regularFields.map((field) => (
                    <WorkflowField key={field.name} field={field} value={values[field.name]} options={options} onChange={(value) => setValues((current) => ({ ...current, [field.name]: value }))} />
                  ))}
                </div>
                <div className="auto-detect-tag">
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>smart_toy</span>
                  Auto-detect source language available
                </div>
              </div>
            )}

            {/* Options & Flags */}
            {booleanFields.length > 0 && (
              <div className="form-section">
                <div className="form-section-title">
                  <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>toggle_on</span>
                  Pipeline Options & Flags
                </div>
                <div className="field-grid-toggles">
                  {booleanFields.map((field) => (
                    <WorkflowField key={field.name} field={field} value={values[field.name]} options={options} onChange={(value) => setValues((current) => ({ ...current, [field.name]: value }))} />
                  ))}
                </div>
              </div>
            )}

            {/* Wizard Footer Action Bar */}
            <div className="wizard-footer">
              <button className="cancel-btn" type="button">Cancel</button>
              <button type="submit" className="next-step-btn">
                Queue job
                <span className="material-symbols-outlined" aria-hidden="true" style={{ fontSize: '18px' }}>arrow_forward</span>
              </button>
            </div>
          </form>
        </div>
      </div>
    </section>
  )
}

function WorkflowField({ field, value, options, onChange }: { field: SchemaFieldDefinition; value: JsonValue; options: OptionsResponse; onChange(value: JsonValue): void }) {
  const label = field.label ?? field.name
  if (field.type === 'boolean') return <label className="field-boolean"><input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />{label}</label>
  if (field.type === 'select') {
    const choices = normalizeOptions(field.options ?? options[field.options_key ?? field.name])
    return <label className="field-select">{label}<select value={String(value ?? '')} onChange={(e) => onChange(e.target.value)}>{choices.map((choice) => <option key={choice.value} value={choice.value}>{choice.label}</option>)}</select></label>
  }
  return <label className="field-input">{label}<input type={field.type === 'number' ? 'number' : 'text'} value={String(value ?? '')} onChange={(e) => onChange(field.type === 'number' ? Number(e.target.value) : e.target.value)} /></label>
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
