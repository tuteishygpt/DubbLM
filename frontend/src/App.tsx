import { useEffect, useRef, useState } from 'react'
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

interface ProjectSummary {
  name: string
  segment_count: number
  has_video: boolean
  has_transcription: boolean
  job_id: string | null
}

interface Job { id: string; status: string }

export default function App({ client = apiClient }: { client?: ApiClient }) {
  const [view, setView] = useState<View>('HukFlow Studio')
  const [config, setConfig] = useState<ConfigResponse>()
  const [error, setError] = useState('')

  // ── Open Project sidebar state ────────────────────────────────────────────
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [showProjectDropdown, setShowProjectDropdown] = useState(false)
  const [openingProject, setOpeningProject] = useState<string | null>(null)
  const [openProjectError, setOpenProjectError] = useState('')
  const [pendingOpenJobId, setPendingOpenJobId] = useState<string | null>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    client.get<ConfigResponse>('/api/config').then(setConfig).catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : String(reason))
    })
  }, [client])

  // Load project list for Open Project dropdown
  useEffect(() => {
    client.get<{ projects?: ProjectSummary[] }>('/api/projects')
      .then((res) => {
        const list = Array.isArray(res?.projects) ? res.projects : []
        setProjects(list)
      })
      .catch(() => {/* ignore if endpoint not available */})
  }, [client])

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!showProjectDropdown) return
    const handle = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowProjectDropdown(false)
      }
    }
    document.addEventListener('mousedown', handle)
    return () => document.removeEventListener('mousedown', handle)
  }, [showProjectDropdown])

  const handleOpenProjectFromSidebar = async (projectName: string) => {
    setShowProjectDropdown(false)
    setOpeningProject(projectName)
    setOpenProjectError('')
    try {
      const res = await client.post<{ project_name: string; job: Job }>(
        `/api/projects/${encodeURIComponent(projectName)}/open`,
      )
      if (res?.job?.id) {
        setPendingOpenJobId(res.job.id)
        setView('HukFlow Studio')
      }
    } catch (err) {
      setOpenProjectError(err instanceof Error ? err.message : 'Failed to open project')
      setTimeout(() => setOpenProjectError(''), 4000)
    } finally {
      setOpeningProject(null)
    }
  }

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

            {/* ── Action Buttons ── */}
            <button
              className="new-clip-btn"
              type="button"
              onClick={() => setView('Workflow')}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>add</span>
              New Clip
            </button>

            {/* Open Project button */}
            <div className="open-project-btn-wrap" ref={dropdownRef}>
              <button
                className="open-project-btn"
                type="button"
                disabled={openingProject !== null}
                onClick={() => setShowProjectDropdown((v) => !v)}
                title="Open an existing project from prj/"
              >
                {openingProject ? (
                  <span className="material-symbols-outlined spin-anim" style={{ fontSize: '16px' }}>sync</span>
                ) : (
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>folder_open</span>
                )}
                {openingProject ? `Opening…` : 'Open Project'}
                <span className="material-symbols-outlined" style={{ fontSize: '14px', marginLeft: 'auto' }}>
                  {showProjectDropdown ? 'expand_less' : 'expand_more'}
                </span>
              </button>

              {showProjectDropdown && (
                <div className="open-project-dropdown" role="listbox" aria-label="Select project to open">
                  {projects.length === 0 ? (
                    <div className="open-project-empty">No projects found in prj/</div>
                  ) : (
                    projects.map((p) => (
                      <button
                        key={p.name}
                        type="button"
                        className="open-project-item"
                        role="option"
                        onClick={() => void handleOpenProjectFromSidebar(p.name)}
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: '16px', color: '#d0bcff', flexShrink: 0 }}>
                          {p.has_video ? 'movie' : 'folder'}
                        </span>
                        <span className="open-project-item-name">{p.name}</span>
                        {p.segment_count > 0 && (
                          <span className="open-project-item-meta">{p.segment_count} seg</span>
                        )}
                        {p.job_id && (
                          <span className="material-symbols-outlined open-project-item-linked" title="Already opened" style={{ fontSize: '14px' }}>
                            link
                          </span>
                        )}
                      </button>
                    ))
                  )}
                </div>
              )}

              {openProjectError && (
                <div className="open-project-error" role="alert">{openProjectError}</div>
              )}
            </div>

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
          {view === 'HukFlow Studio' && (
            <HukFlowStudioView
              client={client}
              onNavigate={setView}
              pendingOpenJobId={pendingOpenJobId}
              onPendingOpenJobConsumed={() => setPendingOpenJobId(null)}
            />
          )}

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
