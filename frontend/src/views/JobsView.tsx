import { useEffect, useState } from 'react'
import type { ApiClient, ConnectionState } from '../api/types'

interface Job { id: string; status: string; result_url?: string; report_url?: string; artifacts?: Array<{ name: string; url: string }> }

export function JobsView({ client }: { client: ApiClient }) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [connection, setConnection] = useState<ConnectionState>('connected')
  const [error, setError] = useState('')

  useEffect(() => {
    let subscriptions: Array<() => void> = []
    client.get<{ jobs: Job[] }>('/api/jobs').then(({ jobs: loaded }) => {
      setJobs(loaded)
      subscriptions = loaded.filter((job) => job.status === 'queued' || job.status === 'running').map((job) => client.subscribeJobEvents(job.id, (event) => {
        const status = event.data.status
        if (event.type === 'status' && typeof status === 'string') setJobs((current) => current.map((item) => item.id === event.job_id ? { ...item, status } : item))
        const message = event.data.message
        if (event.type === 'log' && typeof message === 'string') setLogs((current) => [...current, message])
      }, { onState: setConnection, onError: (reason) => setError(reason.message) }))
    }).catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)))
    return () => subscriptions.forEach((close) => close())
  }, [client])

  return <section>
    <h1>Jobs</h1>
    {error && <p role="alert">{error}</p>}
    <p role="status">{connection === 'reconnecting' ? 'Reconnecting…' : 'Live updates connected'}</p>
    <table><thead><tr><th>Job</th><th>Status</th><th>Outputs</th></tr></thead><tbody>{jobs.map((job) => <tr key={job.id} data-job-id={job.id}><td>{job.id}</td><td>{job.status}</td><td>
      {job.result_url && <a href={job.result_url}>Result</a>} {' '}
      {job.report_url && <a href={job.report_url}>Report</a>} {' '}
      {job.artifacts?.map((artifact) => <a key={artifact.url} href={artifact.url}>{artifact.name}</a>)}
    </td></tr>)}</tbody></table>
    <h2>Logs</h2><ol>{logs.map((log, index) => <li key={`${index}-${log}`}>{log}</li>)}</ol>
  </section>
}
