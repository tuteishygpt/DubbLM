import { useEffect, useState } from 'react'
import type { ApiClient } from '../api/types'

interface Job { id: string; status: string }
interface SegmentAudio { id: string; name: string; url: string }
interface Segment {
  segment_id: string; speaker: string; start: number; end: number; text: string; translation: string
  synthesized_text: string; style_prompt: string; audio: SegmentAudio | null
}
interface TextDocument { revision: string; source: string; segments: Segment[] }
type EditableText = 'speaker' | 'text' | 'translation' | 'synthesized_text' | 'style_prompt'

export function DubbingTextsView({ client }: { client: ApiClient }) {
  const [jobs, setJobs] = useState<Job[]>([]); const [jobId, setJobId] = useState(''); const [document, setDocument] = useState<TextDocument>()
  const [dirty, setDirty] = useState(false); const [selected, setSelected] = useState(''); const [error, setError] = useState('')
  useEffect(() => { client.get<{ jobs: Job[] }>('/api/jobs').then((value) => setJobs(Array.isArray(value.jobs) ? value.jobs : [])).catch(showError) }, [client])
  useEffect(() => { if (!jobId) return; setError(''); client.get<TextDocument>(`/api/jobs/${encodeURIComponent(jobId)}/dubbing-texts`).then((value) => { setDocument(value); setDirty(false); setSelected('') }).catch(showError) }, [client, jobId])

  function showError(reason: unknown) { setError(reason instanceof Error ? reason.message : String(reason)) }
  function edit(id: string, key: EditableText, value: string) {
    setDocument((current) => current ? { ...current, segments: current.segments.map((segment) => segment.segment_id === id ? { ...segment, [key]: value } : segment) } : current); setDirty(true)
  }
  async function save() {
    if (!document) return
    try {
      const result = await client.put<TextDocument>(`/api/jobs/${encodeURIComponent(jobId)}/dubbing-texts`, {
        revision: document.revision,
        segments: document.segments.map(({ audio: _audio, ...segment }) => segment),
      })
      setDocument(result); setDirty(false)
    } catch (reason) { showError(reason) }
  }
  async function regenerate() {
    const segment = document?.segments.find((item) => item.segment_id === selected); if (!segment || !document) return
    try {
      const result = await client.post<{ revision: string; segment: Segment }>(`/api/jobs/${encodeURIComponent(jobId)}/dubbing-texts/${encodeURIComponent(segment.segment_id)}/regenerate`, { revision: document.revision, synthesized_text: segment.synthesized_text })
      setDocument((current) => current ? {
        ...current,
        revision: result.revision,
        segments: current.segments.map((item) => item.segment_id === segment.segment_id
          ? { ...item, synthesized_text: result.segment.synthesized_text, audio: result.segment.audio }
          : item),
      } : current)
    } catch (reason) { showError(reason) }
  }

  return <section><h1>Dubbing Texts</h1>{error && <p role="alert">{error}</p>}
    <label>Job<select value={jobId} onChange={(e) => setJobId(e.target.value)}><option value="">Select a job</option>{jobs.map((job) => <option key={job.id} value={job.id}>{job.id} · {job.status}</option>)}</select></label>
    {document && <><p>{capitalize(document.source)}</p>{dirty && <p role="status">Unsaved changes</p>}
      <div className="actions"><button type="button" disabled={!dirty} onClick={save}>Save changes</button><button type="button" disabled={!selected} onClick={regenerate}>Regenerate selected</button></div>
      <table><thead><tr><th>Select</th><th>Timing</th><th>Speaker</th><th>Original</th><th>Translated</th><th>Synthesized</th><th>Style instructions</th><th>Audio</th></tr></thead><tbody>{document.segments.map((segment) => <tr key={segment.segment_id} data-testid={`segment-${segment.segment_id}`}>
        <td><input aria-label={`Select ${segment.segment_id}`} type="radio" name="selected-segment" checked={selected === segment.segment_id} onChange={() => setSelected(segment.segment_id)} /></td>
        <td>{segment.start}–{segment.end}</td>
        <td><input aria-label={`Speaker ${segment.segment_id}`} value={segment.speaker} onChange={(e) => edit(segment.segment_id, 'speaker', e.target.value)} /></td>
        <td><textarea aria-label={`Original text ${segment.segment_id}`} value={segment.text} onChange={(e) => edit(segment.segment_id, 'text', e.target.value)} /></td>
        <td><textarea aria-label={`Translated text ${segment.segment_id}`} value={segment.translation} onChange={(e) => edit(segment.segment_id, 'translation', e.target.value)} /></td>
        <td><textarea aria-label={`Synthesized text ${segment.segment_id}`} value={segment.synthesized_text} onChange={(e) => edit(segment.segment_id, 'synthesized_text', e.target.value)} /></td>
        <td><textarea aria-label={`Style instructions ${segment.segment_id}`} value={segment.style_prompt} onChange={(e) => edit(segment.segment_id, 'style_prompt', e.target.value)} /></td>
        <td>{segment.audio ? <>{segment.audio.name} <a href={segment.audio.url}>Play audio</a></> : 'Missing audio'}</td>
      </tr>)}</tbody></table>
    </>}
  </section>
}

function capitalize(value: string) { return value ? value[0].toUpperCase() + value.slice(1) : value }
