import { useEffect, useState } from 'react'
import { apiClient } from './api/client'
import type { ApiClient, ConfigResponse } from './api/types'
import { JobsView } from './views/JobsView'
import { WorkflowView } from './views/WorkflowView'

const views = ['Workflow', 'Jobs', 'Settings', 'Voice Profiles', 'Dubbing Texts'] as const
type View = (typeof views)[number]

export default function App({ client = apiClient }: { client?: ApiClient }) {
  const [view, setView] = useState<View>('Workflow')
  const [config, setConfig] = useState<ConfigResponse>()
  const [error, setError] = useState('')

  useEffect(() => {
    client.get<ConfigResponse>('/api/config').then(setConfig).catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : String(reason))
    })
  }, [client])

  return (
    <div className="app-shell">
      <header><strong>DubbLM</strong></header>
      <nav aria-label="Primary">
        {views.map((item) => (
          <button key={item} type="button" aria-current={item === view ? 'page' : undefined} onClick={() => setView(item)}>
            {item}
          </button>
        ))}
      </nav>
      <main>
        {!config && !error && <p role="status">Loading application…</p>}
        {error && <p role="alert">{error}</p>}
        {config && view === 'Workflow' && <WorkflowView client={client} config={config} />}
        {config && view === 'Jobs' && <JobsView client={client} />}
        {config && !['Workflow', 'Jobs'].includes(view) && <h1>{view}</h1>}
      </main>
    </div>
  )
}
