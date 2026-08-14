import { useEffect, useState } from 'react'
import type { ApiClient } from '../api/types'

interface Job {
  id: string
  status: string
}

interface TranscriptItem {
  id: string
  speaker: string
  speakerColor: 'secondary' | 'tertiary'
  timeRange: string
  status: 'completed' | 'pending'
  sourceEn: string
  dubbedBe: string
  isEditable?: boolean
}

const DEFAULT_TRANSCRIPT: TranscriptItem[] = [
  {
    id: 'seg-1',
    speaker: 'Speaker 1',
    speakerColor: 'secondary',
    timeRange: '00:12 - 00:18',
    status: 'completed',
    sourceEn: 'The rapid advancement of these models has completely shifted our paradigm.',
    dubbedBe: 'Хуткае развіццё гэтых мадэляў цалкам змяніла нашу парадыгму.',
    isEditable: false,
  },
  {
    id: 'seg-2',
    speaker: 'Speaker 2',
    speakerColor: 'tertiary',
    timeRange: '18:15 - 20:15',
    status: 'pending',
    sourceEn: 'We need to consider the ethical implications before deployment.',
    dubbedBe: 'Мы павінны ўлічваць этычныя наступствы перад разгортваннем.',
    isEditable: true,
  },
]

export function HukFlowStudioView({ client }: { client?: ApiClient }) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedJobId, setSelectedJobId] = useState<string>('')
  const [transcript, setTranscript] = useState<TranscriptItem[]>(DEFAULT_TRANSCRIPT)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [activeTool, setActiveTool] = useState<'razor' | 'sync' | 'clone' | null>(null)
  const [zoomLevel, setZoomLevel] = useState<number>(50)
  const [mutedTracks, setMutedTracks] = useState<Record<string, boolean>>({ v1: false, v2: false, a1: true })
  const [searchQuery, setSearchQuery] = useState<string>('')
  const [isRegenerating, setIsRegenerating] = useState<boolean>(false)
  const [statusMessage, setStatusMessage] = useState<string>('')

  useEffect(() => {
    if (!client) return
    client.get<{ jobs: Job[] }>('/api/jobs')
      .then((res) => {
        if (Array.isArray(res.jobs)) {
          setJobs(res.jobs)
          if (res.jobs.length > 0 && !selectedJobId) {
            setSelectedJobId(res.jobs[0].id)
          }
        }
      })
      .catch(() => {
        // Fallback to default demo mode
      })
  }, [client, selectedJobId])

  const activeSegment = transcript.find((item) => item.isEditable) ?? transcript[0]

  const handleDubbedChange = (id: string, text: string) => {
    setTranscript((prev) =>
      prev.map((item) => (item.id === id ? { ...item, dubbedBe: text } : item))
    )
  }

  const handleRegenerateAudio = async (id: string) => {
    setIsRegenerating(true)
    setStatusMessage('Regenerating audio model inference...')
    try {
      if (client && selectedJobId) {
        await client.post(`/api/jobs/${encodeURIComponent(selectedJobId)}/dubbing-texts/${encodeURIComponent(id)}/regenerate`, {
          synthesized_text: activeSegment.dubbedBe,
        })
      } else {
        await new Promise((resolve) => setTimeout(resolve, 800))
      }
      setStatusMessage('Audio synthesized successfully.')
    } catch (err) {
      setStatusMessage(err instanceof Error ? err.message : 'Audio synthesis complete.')
    } finally {
      setIsRegenerating(false)
      setTimeout(() => setStatusMessage(''), 3000)
    }
  }

  const toggleTrackMute = (trackId: string) => {
    setMutedTracks((prev) => ({ ...prev, [trackId]: !prev[trackId] }))
  }

  const filteredTranscript = transcript.filter(
    (item) =>
      item.sourceEn.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.dubbedBe.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.speaker.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <section className="studio-view-container">
      <h1 className="sr-only">HukFlow Studio</h1>

      {/* Main Content Workspace: Split Screen */}
      <div className="studio-workspace">
        {/* Left Pane: Localization Transcript */}
        <div className="studio-left-pane">
          <div className="pane-header">
            <div className="flex items-center gap-2">
              <h3 className="pane-title">Localization Transcript</h3>
              {jobs.length > 0 && (
                <select
                  aria-label="Job selector"
                  className="studio-job-select"
                  value={selectedJobId}
                  onChange={(e) => setSelectedJobId(e.target.value)}
                >
                  {jobs.map((j) => (
                    <option key={j.id} value={j.id}>
                      {j.id} ({j.status})
                    </option>
                  ))}
                </select>
              )}
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
              <button className="icon-btn" title="Filter list" type="button">
                <span className="material-symbols-outlined">filter_list</span>
              </button>
            </div>
          </div>

          {statusMessage && <div className="studio-status-banner" role="status">{statusMessage}</div>}

          <div className="transcript-scroll-area">
            {filteredTranscript.map((item) => (
              <div
                key={item.id}
                className={`transcript-card ${item.isEditable ? 'active-card' : ''}`}
                data-testid={`transcript-card-${item.id}`}
              >
                <div className={`card-accent-line ${item.speakerColor}`}></div>
                <div className="card-header">
                  <div className="speaker-info">
                    <span className={`speaker-dot ${item.speakerColor}`}></span>
                    <span className="speaker-name">{item.speaker}</span>
                    <span className="time-badge">{item.timeRange}</span>
                  </div>
                  <div className="card-status">
                    {item.status === 'completed' ? (
                      <span className="material-symbols-outlined status-completed" title="Completed">
                        check_circle
                      </span>
                    ) : (
                      <span className="material-symbols-outlined status-pending" title="Pending">
                        pending
                      </span>
                    )}
                  </div>
                </div>

                <div className="card-grid">
                  <div className="source-col">
                    <p className="col-label">Source (EN)</p>
                    <p className="source-text">{item.sourceEn}</p>
                  </div>
                  <div className="dubbed-col">
                    <p className="col-label">Dubbed (BE)</p>
                    {item.isEditable ? (
                      <div className="editable-dubbed-wrapper">
                        <textarea
                          className="dubbed-textarea"
                          value={item.dubbedBe}
                          onChange={(e) => handleDubbedChange(item.id, e.target.value)}
                          rows={3}
                          aria-label={`Dubbed text ${item.speaker}`}
                        />
                        <button
                          type="button"
                          className={`regenerate-btn ${isRegenerating ? 'spinning' : ''}`}
                          onClick={() => handleRegenerateAudio(item.id)}
                          title="Regenerate Audio"
                          disabled={isRegenerating}
                        >
                          <span className="material-symbols-outlined">autorenew</span>
                        </button>
                      </div>
                    ) : (
                      <div className="dubbed-read-only">{item.dubbedBe}</div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Pane: Video Preview & Player Controls */}
        <div className="studio-right-pane">
          <div className="player-top-bar">
            <span className="program-title">Program: Main_Edit_v2</span>
            <span className="video-specs">1080p | 23.976 fps</span>
          </div>

          <div className="video-preview-wrapper">
            <div className="video-canvas">
              <img
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuCJRRkAqd54AAO2XPNAEVRwFN3eAUaVZlZEwRn5ynAyrLB-nGo1zoGPStK-f64Fr0chvGz1D7VAe8Slqvecf0eYvfBB0zdE3P3KaL46qVW4wHoHViygBFwnfGRGvBIZJJ502Bh0I_SDBlhgw34vb7AZLkVFdAcLKp3ncAzPOlJBq-CKcQWrO0hXOiKNhh9A7pJdevkG72EkEGYNCEiPaxr6HJIF-l7Xk3fUj22b8OxdZ-gV4yePYd19gQ"
                alt="Video Preview"
                className="video-poster"
              />
              <div className="subtitle-overlay">
                <span>{activeSegment.dubbedBe}</span>
              </div>
            </div>
          </div>

          {/* Player Transport Controls */}
          <div className="player-controls-bar">
            <div className="timecode-display">
              <span>00:00:18:15</span>
              <span>00:04:32:00</span>
            </div>

            <div className="player-scrubber">
              <div className="scrubber-track">
                <div className="scrubber-progress" style={{ width: '15%' }}></div>
                <div className="scrubber-handle" style={{ left: '15%' }}></div>
              </div>
            </div>

            <div className="transport-buttons">
              <button type="button" className="transport-btn" title="Skip Previous">
                <span className="material-symbols-outlined">skip_previous</span>
              </button>
              <button type="button" className="transport-btn" title="Fast Rewind">
                <span className="material-symbols-outlined">fast_rewind</span>
              </button>
              <button
                type="button"
                className="play-pause-btn"
                title={isPlaying ? 'Pause' : 'Play'}
                onClick={() => setIsPlaying(!isPlaying)}
              >
                <span className="material-symbols-outlined">
                  {isPlaying ? 'pause' : 'play_arrow'}
                </span>
              </button>
              <button type="button" className="transport-btn" title="Fast Forward">
                <span className="material-symbols-outlined">fast_forward</span>
              </button>
              <button type="button" className="transport-btn" title="Skip Next">
                <span className="material-symbols-outlined">skip_next</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Docked Bottom Timeline */}
      <div className="studio-bottom-timeline">
        {/* Timeline Header Toolbar */}
        <div className="timeline-toolbar">
          <div className="toolbar-tools">
            <button
              type="button"
              className={`tool-btn ${activeTool === 'razor' ? 'active' : ''}`}
              title="Razor Cut"
              onClick={() => setActiveTool(activeTool === 'razor' ? null : 'razor')}
            >
              <span className="material-symbols-outlined">content_cut</span>
            </button>
            <button
              type="button"
              className={`tool-btn ${activeTool === 'sync' ? 'active' : ''}`}
              title="Sync Audio"
              onClick={() => setActiveTool(activeTool === 'sync' ? null : 'sync')}
            >
              <span className="material-symbols-outlined">sync_alt</span>
            </button>
            <button
              type="button"
              className={`tool-btn ${activeTool === 'clone' ? 'active' : ''}`}
              title="Voice Clone"
              onClick={() => setActiveTool(activeTool === 'clone' ? null : 'clone')}
            >
              <span className="material-symbols-outlined">record_voice_over</span>
            </button>
          </div>

          <div className="toolbar-zoom">
            <span className="material-symbols-outlined zoom-icon">zoom_out</span>
            <input
              type="range"
              min="10"
              max="100"
              value={zoomLevel}
              onChange={(e) => setZoomLevel(Number(e.target.value))}
              className="zoom-slider"
              aria-label="Zoom timeline"
            />
            <span className="material-symbols-outlined zoom-icon">zoom_in</span>
          </div>
        </div>

        {/* Timeline Track Rows */}
        <div className="timeline-body">
          {/* Left Track Headers */}
          <div className="track-headers-column">
            <div className="ruler-corner"></div>
            <div className="track-header border-secondary">
              <span className="track-name">V1: Spk 1</span>
              <button
                type="button"
                className="mute-btn"
                onClick={() => toggleTrackMute('v1')}
                title="Toggle Mute"
              >
                <span className="material-symbols-outlined">
                  {mutedTracks.v1 ? 'volume_off' : 'volume_up'}
                </span>
              </button>
            </div>
            <div className="track-header border-tertiary">
              <span className="track-name">V2: Spk 2</span>
              <button
                type="button"
                className="mute-btn"
                onClick={() => toggleTrackMute('v2')}
                title="Toggle Mute"
              >
                <span className="material-symbols-outlined">
                  {mutedTracks.v2 ? 'volume_off' : 'volume_up'}
                </span>
              </button>
            </div>
            <div className="track-header border-music">
              <span className="track-name">A1: Music</span>
              <button
                type="button"
                className="mute-btn"
                onClick={() => toggleTrackMute('a1')}
                title="Toggle Mute"
              >
                <span className="material-symbols-outlined">
                  {mutedTracks.a1 ? 'volume_off' : 'volume_up'}
                </span>
              </button>
            </div>
          </div>

          {/* Timeline Grid & Clips */}
          <div className="timeline-tracks-canvas">
            {/* Time Ruler */}
            <div className="time-ruler">
              <span>00:00:00</span>
              <span>00:00:10</span>
              <span>00:00:20</span>
              <span>00:00:30</span>
              <span>00:00:40</span>
            </div>

            {/* Playhead */}
            <div className="timeline-playhead" style={{ left: '35%' }}>
              <div className="playhead-head"></div>
            </div>

            {/* Track 1 Row */}
            <div className="track-row">
              <div className="clip-block secondary-clip" style={{ left: '5%', width: '25%' }}>
                <svg className="waveform-svg" viewBox="0 0 100 20" preserveAspectRatio="none">
                  <path d="M0,10 L5,2 L10,18 L15,5 L20,15 L25,8 L30,12 L35,4 L40,16 L45,10 L50,6 L55,14 L60,3 L65,17 L70,9 L75,11 L80,5 L85,15 L90,8 L95,12 L100,10" fill="none" stroke="currentColor" strokeWidth="1.5" />
                </svg>
              </div>
            </div>

            {/* Track 2 Row (Active AI Shimmer) */}
            <div className="track-row">
              <div className="clip-block primary-clip ai-shimmer" style={{ left: '32%', width: '22%' }}>
                <svg className="waveform-svg" viewBox="0 0 100 20" preserveAspectRatio="none">
                  <path d="M0,10 L3,14 L6,6 L9,18 L12,10 L15,12 L18,8 L21,15 L24,5 L27,10 L30,14 L33,6 L36,18 L39,10 L42,12 L45,8 L48,15 L51,5 L54,10 L57,14 L60,6 L63,18 L66,10 L69,12 L72,8 L75,15 L78,5 L81,10 L84,14 L87,6 L90,18 L93,10 L96,12 L100,10" fill="none" stroke="currentColor" strokeWidth="1" />
                </svg>
                <span className="clip-subtitle">{activeSegment.dubbedBe}</span>
              </div>
            </div>

            {/* Track 3 Row (Music) */}
            <div className="track-row">
              <div className="clip-block music-clip" style={{ left: '0%', width: '95%' }}>
                <svg className="waveform-svg opacity-50" viewBox="0 0 200 20" preserveAspectRatio="none">
                  <path d="M0,10 Q10,5 20,10 T40,10 T60,10 T80,10 T100,10 T120,10 T140,10 T160,10 T180,10 T200,10" fill="none" stroke="currentColor" strokeWidth="1" />
                </svg>
                <span className="clip-label">Corporate_Bed_01.wav</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
