import React from 'react'
import { SegmentItem } from '../components/studio/SegmentItem'
import { SettingsModal } from '../components/studio/SettingsModal'
import { TimelineCanvas } from '../components/studio/TimelineCanvas'
import { VideoPlayerCanvas } from '../components/studio/VideoPlayerCanvas'
import { StudioProvider, useStudio, RUN_STEPS } from '../contexts/StudioContext'
import type { ApiClient } from '../api/types'
import type { NavigateView, StudioMediaInfo } from '../contexts/StudioContext'
import { createPortal as createPortalDom } from 'react-dom'

function StudioLayout() {
  const studio = useStudio()
  const {
    portalTarget,
    selectedProjectName,
    selectedJobId,
    runDropdownRef,
    showRunDropdown,
    setShowRunDropdown,
    isRunningStep,
    handleRunStep,
    projects,
    jobs,
    handleOpenProject,
    setSelectedJobId,
    setSelectedProjectName,
    dirty,
    handleSave,
    searchQuery,
    setSearchQuery,
    statusMessage,
    loadError,
    scrollAreaRef,
    isLoading,
    visibleSegments,
    currentActive,
    displaySpeakers,
    expandedSources,
  } = studio

  const runMenuNode = (selectedProjectName || selectedJobId) ? (
    <div className="studio-run-menu-wrap" ref={runDropdownRef}>
      <button
        type="button"
        className="studio-run-btn"
        onClick={() => setShowRunDropdown((v: boolean) => !v)}
        disabled={isRunningStep}
        title="Run or resume dubbing pipeline steps"
        aria-haspopup="true"
        aria-expanded={showRunDropdown}
      >
        {isRunningStep ? (
          <span className="material-symbols-outlined spin-anim">sync</span>
        ) : (
          <span className="material-symbols-outlined">bolt</span>
        )}
        <span>{isRunningStep ? 'Starting…' : 'Run Step'}</span>
        <span className="material-symbols-outlined dropdown-arrow">
          {showRunDropdown ? 'expand_less' : 'expand_more'}
        </span>
      </button>

      {showRunDropdown && (
        <div className="run-menu-dropdown" role="menu" aria-label="Pipeline steps">
          <div className="run-menu-header">
            <span className="material-symbols-outlined">settings_suggest</span>
            <span>Pipeline Execution Steps</span>
          </div>
          {RUN_STEPS.map((step) => (
            <button
              key={step.id}
              type="button"
              className="run-menu-item"
              role="menuitem"
              onClick={() => void handleRunStep(step.id)}
            >
              <div className="run-item-icon">
                <span className="material-symbols-outlined">{step.icon}</span>
              </div>
              <div className="run-item-body">
                <div className="run-item-top">
                  <span className="run-item-title">{step.title}</span>
                  <span className="run-item-badge">{step.badge}</span>
                </div>
                <span className="run-item-desc">{step.description}</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  ) : null

  return (
    <section className="studio-view-container">
      <h1 className="sr-only">HukFlow Studio</h1>

      {portalTarget && runMenuNode ? createPortalDom(runMenuNode, portalTarget) : null}

      <div className="studio-workspace">
        <div className="studio-left-pane">
          <div className="pane-header">
            <div className="pane-header-left">
              <h3 className="pane-title">Localization Transcript</h3>
              {projects.length > 0 && (
                <select
                  aria-label="Ready project selector"
                  className="studio-job-select studio-project-select"
                  value={selectedProjectName}
                  onChange={(e) => {
                    const val = e.target.value
                    if (val) handleOpenProject(val)
                  }}
                >
                  <option value="">📁 Open Project (prj/)...</option>
                  {projects.map((p) => (
                    <option key={p.name} value={p.name}>
                      📁 {p.display_name ?? p.name} {p.segment_count > 0 ? `(${p.segment_count} segments)` : ''}
                    </option>
                  ))}
                </select>
              )}
              {jobs.length > 0 && (
                <select
                  aria-label="Job selector"
                  className="studio-job-select"
                  value={selectedJobId}
                  onChange={(e) => {
                    setSelectedJobId(e.target.value)
                    setSelectedProjectName('')
                  }}
                >
                  {jobs.map((j) => (
                    <option key={j.id} value={j.id}>
                      {j.id} ({j.status})
                    </option>
                  ))}
                </select>
              )}
              {dirty && (
                <button type="button" className="save-badge-btn" onClick={handleSave}>
                  <span className="material-symbols-outlined">save</span>
                  Save
                </button>
              )}

              {!portalTarget && runMenuNode}
            </div>

            <div className="pane-actions">
              <div className="search-input-wrapper">
                <span className="material-symbols-outlined search-icon">search</span>
                <input
                  type="text"
                  placeholder="Search transcript..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="search-input"
                  aria-label="Search transcript"
                />
              </div>
              <button className="icon-btn" title="Filter" type="button">
                <span className="material-symbols-outlined">filter_list</span>
              </button>
            </div>
          </div>

          {statusMessage && (
            <div className="studio-status-banner" role="status">{statusMessage}</div>
          )}
          {loadError && (
            <div className="studio-error-banner" role="alert">{loadError}</div>
          )}

          <div className="transcript-scroll-area" ref={scrollAreaRef}>
            {isLoading && (
              <div className="transcript-loading">
                <span className="material-symbols-outlined spin-anim">sync</span>
                <span>Loading dubbing texts…</span>
              </div>
            )}

            {!isLoading && visibleSegments.length === 0 && (
              <div className="transcript-empty" style={{ padding: '24px', textAlign: 'center', color: '#938f99' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '36px', marginBottom: '8px', display: 'block' }}>subtitles_off</span>
                <p>{searchQuery ? 'No matching segments found' : 'No dubbing texts available for this job'}</p>
              </div>
            )}

            {!isLoading &&
              visibleSegments.map((seg, idx: number) => {
                const isActive = seg.segment_id === (currentActive?.segment_id ?? '')
                const speakerIdx = displaySpeakers.indexOf(seg.speaker)
                const isSourceExpanded = Boolean(expandedSources[seg.segment_id])

                return (
                  <SegmentItem
                    key={seg.segment_id}
                    segment={seg}
                    isActive={isActive}
                    speakerIdx={speakerIdx}
                    isSourceExpanded={isSourceExpanded}
                    segmentIndex={idx}
                    totalSegments={visibleSegments.length}
                  />
                )
              })}
          </div>
        </div>

        <div className="studio-right-pane">
          <VideoPlayerCanvas />
        </div>
      </div>

      <div className="studio-bottom-timeline">
        <div className="timeline-floating-zoom" onClick={(e) => e.stopPropagation()}>
          <button
            type="button"
            className="zoom-btn"
            title="Zoom Out"
            onClick={() => studio.setZoomLevel((z: number) => Math.max(10, z - 10))}
            aria-label="Zoom Out"
          >
            <span className="material-symbols-outlined">zoom_out</span>
          </button>
          <input
            type="range"
            min="10"
            max="100"
            value={studio.zoomLevel}
            onChange={(e) => studio.setZoomLevel(Number(e.target.value))}
            className="zoom-slider"
            aria-label="Zoom timeline"
          />
          <button
            type="button"
            className="zoom-btn"
            title="Zoom In"
            onClick={() => studio.setZoomLevel((z: number) => Math.min(100, z + 10))}
            aria-label="Zoom In"
          >
            <span className="material-symbols-outlined">zoom_in</span>
          </button>
        </div>

        <TimelineCanvas />
      </div>

      <SettingsModal />
    </section>
  )
}

export function HukFlowStudioView(props: {
  client?: ApiClient
  onNavigate?: (view: NavigateView) => void
  pendingOpenJobId?: string | null
  onPendingOpenJobConsumed?: () => void
  onJobStarted?: (jobId: string, projectName?: string) => void
  onActiveMediaChange?: (info: StudioMediaInfo) => void
  showSettingsPanel?: boolean
  onCloseSettingsPanel?: () => void
}) {
  return (
    <StudioProvider {...props}>
      <StudioLayout />
    </StudioProvider>
  )
}
