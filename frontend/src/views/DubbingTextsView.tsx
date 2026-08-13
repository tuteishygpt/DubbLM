import { useEffect, useState } from 'react'
import type { ApiClient } from '../api/types'

interface Job { id: string; status: string }
interface Segment {
  id: string; speaker: string; start: number; end: number; original_text: string; translated_text: string
  synthesized_text: string; audio_path: string | null; audio_url: string | null
}
interface TextDocument { revision: string; source: string; cached: boolean; segments: Segment[] }

export function DubbingTextsView({ client }: { client: ApiClient }) {
  const [jobs, setJobs] = useState<Job[]>([]); const [jobId, setJobId] = useState(''); const [document, setDocument] = useState<TextDocument>()
  const [dirty, setDirty] = useState(false); const [selected, setSelected] = useState(''); const [error, setError] = useState('')
  useEffect(() => { client.get<{ jobs: Job[] }>('/api/jobs').then((value) => setJobs(Array.isArray(value.jobs) ? value.jobs : [])).catch((reason) => setError(String(reason))) }, [client])
  useEffect(() => { if (!jobId) return; setError(''); client.get<TextDocument>(`/api/jobs/${encodeURIComponent(jobId)}/dubbing-texts`).then((value) => { setDocument(value); setDirty(false); setSelected('') }).catch((reason) => setError(String(reason))) }, [client, jobId])

  function edit(id: string, key: keyof Pick<Segment, 'speaker' | 'original_text' | 'translated_text' | 'synthesized_text'>, value: string) {
    setDocument((current) => current ? { ...current, segments: current.segments.map((segment) => segment.id === id ? { ...segment, [key]: value } : segment) } : current); setDirty(true)
  }
  async function save() {
    if (!document) return
    try { const result = await client.put<{ revision?: string }>(`/api/jobs/${encodeURIComponent(jobId)}/dubbing-texts`, { revision: document.revision, segments: document.segments }); setDocument({ ...document, revision: result.revision ?? document.revision }); setDirty(false) }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }
  async function regenerate() {
    const segment = document?.segments.find((item) => item.id === selected); if (!segment) return
    try {
      const result = await client.post<{ segment: Segment }>(`/api/jobs/${encodeURIComponent(jobId)}/dubbing-texts/${encodeURIComponent(segment.id)}/regenerate`, { synthesized_text: segment.synthesized_text })
      setDocument((current) => current ? { ...current, segments: current.segments.map((item) => item.id === segment.id ? { ...item, ...result.segment } : item) } : current)
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }

  return <section><h1>Dubbing Texts</h1>{error && <p role="alert">{error}</p>}
    <label>Job<select value={jobId} onChange={(e) => setJobId(e.target.value)}><option value="">Select a job</option>{jobs.map((job) => <option key={job.id} value={job.id}>{job.id} · {job.status}</option>)}</select></label>
    {document && <><p>{capitalize(document.source)} · {document.cached ? 'cached' : 'not cached'}</p>{dirty && <p role="status">Unsaved changes</p>}
      <div className="actions"><button type="button" disabled={!dirty} onClick={save}>Save changes</button><button type="button" disabled={!selected} onClick={regenerate}>Regenerate selected</button></div>
      <table><thead><tr><th>Select</th><th>Timing</th><th>Speaker</th><th>Original</th><th>Translated</th><th>Synthesized</th><th>Audio</th></tr></thead><tbody>{document.segments.map((segment) => <tr key={segment.id} data-testid={`segment-${segment.id}`}>
        <td><input aria-label={`Select ${segment.id}`} type="radio" name="selected-segment" checked={selected === segment.id} onChange={() => setSelected(segment.id)} /></td>
        <td>{segment.start}–{segment.end}</td>
        <td><input aria-label={`Speaker ${segment.id}`} value={segment.speaker} onChange={(e) => edit(segment.id, 'speaker', e.target.value)} /></td>
        <td><textarea aria-label={`Original text ${segment.id}`} value={segment.original_text} onChange={(e) => edit(segment.id, 'original_text', e.target.value)} /></td>
        <td><textarea aria-label={`Translated text ${segment.id}`} value={segment.translated_text} onChange={(e) => edit(segment.id, 'translated_text', e.target.value)} /></td>
        <td><textarea aria-label={`Synthesized text ${segment.id}`} value={segment.synthesized_text} onChange={(e) => edit(segment.id, 'synthesized_text', e.target.value)} /></td>
        <td>{segment.audio_path ?? 'Missing audio'} {segment.audio_url && <a href={segment.audio_url}>Play audio</a>}</td>
      </tr>)}</tbody></table>
    </>}
  </section>
}

function capitalize(value: string) { return value ? value[0].toUpperCase() + value.slice(1) : value }
