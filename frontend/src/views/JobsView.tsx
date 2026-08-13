import { useEffect, useState } from 'react'
import type { ApiClient, ConnectionState, JsonValue, SseEvent } from '../api/types'

interface JobFile { id: string; name: string; kind: string; size: number }
interface Job { id: string; status: string; files: JobFile[] }

export function JobsView({ client }: { client: ApiClient }) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [connection, setConnection] = useState<ConnectionState>('connected')
  const [error, setError] = useState('')

  useEffect(() => {
    let subscriptions: Array<() => void> = []
    client.get<{ jobs: Job[] }>('/api/jobs').then(({ jobs: loaded }) => {
      const loadedJobs = Array.isArray(loaded) ? loaded.map((job) => ({ ...job, files: Array.isArray(job.files) ? job.files : [] })) : []
      setJobs(loadedJobs)
      subscriptions = loadedJobs.filter((job) => job.status === 'queued' || job.status === 'running').map((job) => client.subscribeJobEvents(job.id, receive, { onState: setConnection, onError: (reason) => setError(reason.message) }))
    }).catch(showError)
    return () => subscriptions.forEach((close) => close())

    function showError(reason: unknown) { setError(reason instanceof Error ? reason.message : String(reason)) }
    function receive(event: SseEvent) {
      if (event.type === 'log') appendLog(event.data.message)
      if (event.type === 'state') updateStatus(event.job_id, event.data.status)
      if (event.type === 'file') addFile(event.job_id, event.data)
      if (event.type === 'error') showEventError(event.data)
      if (event.type === 'snapshot') {
        updateStatus(event.job_id, event.data.status)
        replaceFiles(event.job_id, event.data.files)
        const tail = event.data.log_tail
        if (Array.isArray(tail)) for (const item of tail) if (isRecord(item)) appendLog(item.message)
        const snapshotError = event.data.error
        if (isRecord(snapshotError)) showEventError(snapshotError)
      }
    }
    function updateStatus(jobId: string, value: JsonValue | undefined) {
      if (typeof value === 'string') setJobs((current) => current.map((job) => job.id === jobId ? { ...job, status: value } : job))
    }
    function appendLog(value: JsonValue | undefined) { if (typeof value === 'string') setLogs((current) => [...current, value]) }
    function showEventError(value: Record<string, JsonValue>) { if (typeof value.message === 'string') setError(value.message) }
    function addFile(jobId: string, value: Record<string, JsonValue>) {
      const file = parseFile(value)
      if (file) setJobs((current) => current.map((job) => job.id === jobId ? { ...job, files: [...job.files.filter((item) => item.id !== file.id), file] } : job))
    }
    function replaceFiles(jobId: string, value: JsonValue | undefined) {
      if (!Array.isArray(value)) return
      const files = value.map((item) => isRecord(item) ? parseFile(item) : undefined).filter((item): item is JobFile => item !== undefined)
      setJobs((current) => current.map((job) => job.id === jobId ? { ...job, files } : job))
    }
  }, [client])

  return <section>
    <h1>Jobs</h1>
    {error && <p role="alert">{error}</p>}
    <p role="status">{connection === 'reconnecting' ? 'Reconnecting…' : 'Live updates connected'}</p>
    <table><thead><tr><th>Job</th><th>Status</th><th>Outputs</th></tr></thead><tbody>{jobs.map((job) => <tr key={job.id} data-job-id={job.id}><td>{job.id}</td><td>{job.status}</td><td>
      {job.files.map((file) => <a key={file.id} href={`/api/jobs/${encodeURIComponent(job.id)}/files/${encodeURIComponent(file.id)}`}>{file.name}</a>)}
    </td></tr>)}</tbody></table>
    <h2>Logs</h2><ol>{logs.map((log, index) => <li key={`${index}-${log}`}>{log}</li>)}</ol>
  </section>
}

function isRecord(value: JsonValue): value is Record<string, JsonValue> { return typeof value === 'object' && value !== null && !Array.isArray(value) }
function parseFile(value: Record<string, JsonValue>): JobFile | undefined {
  return typeof value.id === 'string' && typeof value.name === 'string' && typeof value.kind === 'string' && typeof value.size === 'number'
    ? { id: value.id, name: value.name, kind: value.kind, size: value.size }
    : undefined
}
