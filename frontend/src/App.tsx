import { useEffect, useState } from 'react'
import { apiClient } from './api/client'
import type { ApiClient, ConfigResponse } from './api/types'
import { JobsView } from './views/JobsView'
import { WorkflowView } from './views/WorkflowView'
import { SettingsView } from './views/SettingsView'
import { VoicesView } from './views/VoicesView'
import { DubbingTextsView } from './views/DubbingTextsView'
import { HukFlowStudioView } from './views/HukFlowStudioView'

const views = ['HukFlow Studio', 'Workflow', 'Jobs', 'Settings', 'Voice Profiles', 'Dubbing Texts'] as const
type View = (typeof views)[number]

export default function App({ client = apiClient }: { client?: ApiClient }) {
  const [view, setView] = useState<View>('HukFlow Studio')
  const [config, setConfig] = useState<ConfigResponse>()
  const [error, setError] = useState('')

  useEffect(() => {
    client.get<ConfigResponse>('/api/config').then(setConfig).catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : String(reason))
    })
  }, [client])

  const isStudio = view === 'HukFlow Studio'

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

        {/* Studio: show project info + nav inline in header */}
        {isStudio && (
          <nav className="header-studio-nav" aria-label="Studio navigation">
            {views.filter(v => v !== 'HukFlow Studio').map((item) => (
              <button
                key={item}
                type="button"
                className="header-nav-btn"
                onClick={() => setView(item)}
              >
                {item}
              </button>
            ))}
          </nav>
        )}

        <div className="header-actions">
          <button className="header-icon-btn" title="cloud_done">
            <span className="material-symbols-outlined">cloud_done</span>
          </button>
          <button className="header-icon-btn" title="settings">
            <span className="material-symbols-outlined">settings</span>
          </button>
          {!isStudio && <button className="export-btn">Export</button>}
          {isStudio && (
            <>
              <button className="header-nav-btn-secondary" type="button">Share</button>
              <button className="export-btn">Export</button>
            </>
          )}
          <div className="user-avatar" title="User profile">
            <span className="material-symbols-outlined" style={{ fontSize: '18px', color: '#cbc3d7' }}>person</span>
          </div>
        </div>
      </header>

      {/* Main Body Grid */}
      <div className={`app-body${isStudio ? ' app-body--studio' : ''}`}>
        {/* Left Sidebar Navigation — hidden in Studio mode */}
        {!isStudio && (
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
        )}

        {/* Main Canvas Workspace */}
        <main className={isStudio ? 'main--studio' : ''}>
          {/* Studio renders immediately with demo data — no backend required */}
          {view === 'HukFlow Studio' && <HukFlowStudioView client={client} />}

          {/* Other views need config from backend */}
          {view !== 'HukFlow Studio' && !config && !error && <p role="status">Loading application…</p>}
          {view !== 'HukFlow Studio' && error && <p role="alert">{error}</p>}
          {view !== 'HukFlow Studio' && config && view === 'Workflow' && <WorkflowView client={client} config={config} />}
          {view !== 'HukFlow Studio' && config && view === 'Jobs' && <JobsView client={client} />}
          {view !== 'HukFlow Studio' && config && view === 'Settings' && <SettingsView client={client} config={config} />}
          {view !== 'HukFlow Studio' && config && view === 'Voice Profiles' && <VoicesView client={client} />}
          {view !== 'HukFlow Studio' && config && view === 'Dubbing Texts' && <DubbingTextsView client={client} />}
        </main>
      </div>
    </div>
  )
}
