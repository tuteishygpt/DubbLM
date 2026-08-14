import { useEffect, useState } from 'react'
import { apiClient } from './api/client'
import type { ApiClient, ConfigResponse } from './api/types'
import { JobsView } from './views/JobsView'
import { WorkflowView } from './views/WorkflowView'
import { SettingsView } from './views/SettingsView'
import { VoicesView } from './views/VoicesView'
import { DubbingTextsView } from './views/DubbingTextsView'

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
      {/* Stitch TopNavBar Header */}
      <header>
        <div className="studio-brand">
          <div className="studio-brand-logo">H</div>
          <div>
            <strong>HukFlow</strong>
          </div>
          <span className="studio-tagline">AI DUBBING</span>
        </div>
        <div className="header-actions">
          <button className="header-icon-btn" title="cloud_done">
            <span className="material-symbols-outlined">cloud_done</span>
          </button>
          <button className="header-icon-btn" title="settings">
            <span className="material-symbols-outlined">settings</span>
          </button>
          <button className="export-btn">Export</button>
          <div className="user-avatar" title="User profile">
            <span className="material-symbols-outlined" style={{ fontSize: '18px', color: '#cbc3d7' }}>person</span>
          </div>
        </div>
      </header>

      {/* Main Body Grid */}
      <div className="app-body">
        {/* Left Sidebar Navigation */}
        <aside className="sidebar">
          <div className="project-summary-card">
            <div className="project-thumb">
              <span className="material-symbols-outlined" style={{ color: '#d0bcff' }}>movie</span>
            </div>
            <div className="project-details">
              <span className="project-name">Project Alpha</span>
              <span className="project-status">Localizing: EN to BE</span>
            </div>
          </div>

          <button className="new-clip-btn" type="button">
            <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>add</span>
            New Clip
          </button>

          <nav aria-label="Primary">
            {views.map((item) => (
              <button key={item} type="button" data-view={item} aria-current={item === view ? 'page' : undefined} onClick={() => setView(item)}>
                {item}
              </button>
            ))}
          </nav>

          <div className="sidebar-footer">
            <a href="#">
              <span className="material-symbols-outlined">help_outline</span>
              Help
            </a>
            <a href="#">
              <span className="material-symbols-outlined">chat_bubble_outline</span>
              Feedback
            </a>
          </div>
        </aside>

        {/* Main Canvas Workspace */}
        <main>
          {!config && !error && <p role="status">Loading application…</p>}
          {error && <p role="alert">{error}</p>}
          {config && view === 'Workflow' && <WorkflowView client={client} config={config} />}
          {config && view === 'Jobs' && <JobsView client={client} />}
          {config && view === 'Settings' && <SettingsView client={client} config={config} />}
          {config && view === 'Voice Profiles' && <VoicesView client={client} />}
          {config && view === 'Dubbing Texts' && <DubbingTextsView client={client} />}
        </main>
      </div>
    </div>
  )
}
