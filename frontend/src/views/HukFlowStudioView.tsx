import { useEffect, useRef, useState } from 'react'
import type { ApiClient } from '../api/types'

// ── API types ─────────────────────────────────────────────────────────────
interface Job { id: string; status: string }

interface JobFile { id: string; name: string; kind: string; size: number; url?: string }

interface SegmentAudio { id: string; name: string; url: string }

interface Segment {
  segment_id: string
  speaker: string
  start: number
  end: number
  text: string
  translation: string
  synthesized_text: string
  style_prompt: string
  audio: SegmentAudio | null
}

interface TextDocument { revision: string; source: string; segments: Segment[] }

const SPEAKER_COLOURS = ['speaker-0', 'speaker-1', 'speaker-2', 'speaker-3'] as const

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function formatTimecode(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  const f = Math.floor((seconds % 1) * 24)
  return `00:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}:${String(f).padStart(2, '0')}`
}

export function HukFlowStudioView({ client }: { client?: ApiClient }) {
  // ── jobs / document state ─────────────────────────────────────────────────
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedJobId, setSelectedJobId] = useState<string>('')
  const [document, setDocument] = useState<TextDocument | null>(null)
  const [segments, setSegments] = useState<Segment[]>([])
  const [revision, setRevision] = useState<string>('')
  const [dirty, setDirty] = useState(false)
  const [loadError, setLoadError] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)

  // ── active segment (selected card) ───────────────────────────────────────
  const [activeSegmentId, setActiveSegmentId] = useState<string>('')

  // ── dynamic video player state ───────────────────────────────────────────
  const videoRef = useRef<HTMLVideoElement>(null)
  const [videoSrc, setVideoSrc] = useState<string>('/video.mp4')
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [videoDuration, setVideoDuration] = useState(60)

  // ── studio tools state ───────────────────────────────────────────────────
  const [activeTool, setActiveTool] = useState<'razor' | 'sync' | 'clone' | null>(null)
  const [zoomLevel, setZoomLevel] = useState(50)
  const [mutedTracks, setMutedTracks] = useState<Record<string, boolean>>({ a1: true })
  const [searchQuery, setSearchQuery] = useState('')
  const [isRegenerating, setIsRegenerating] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')

  // Fallback demo segments if backend is empty
  const displaySegments = segments.length > 0 ? segments : [
    {
      segment_id: 'demo-1',
      speaker: 'Speaker 1',
      start: 0,
      end: 6,
      text: 'The rapid advancement of these models has completely shifted our paradigm.',
      translation: 'Хуткае развіццё гэтых мадэляў цалкам змяніла нашу парадыгму.',
      synthesized_text: '',
      style_prompt: '',
      audio: { id: 'a1', name: 'audio_1.wav', url: '#' }
    },
    {
      segment_id: 'demo-2',
      speaker: 'Speaker 2',
      start: 7,
      end: 15,
      text: 'We need to consider the ethical implications before deployment.',
      translation: 'Мы павінны ўлічваць этычныя наступствы перад разгортваннем.',
      synthesized_text: '',
      style_prompt: '',
      audio: null
    }
  ]

  const displaySpeakers = [...new Set(displaySegments.map((s) => s.speaker))]

  // ── load jobs ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!client) return
    client.get<{ jobs: Job[] }>('/api/jobs')
      .then((res) => {
        const list = Array.isArray(res.jobs) ? res.jobs : []
        setJobs(list)
        if (list.length > 0) setSelectedJobId((prev) => prev || list[0].id)
      })
      .catch(() => {/* demo mode */})
  }, [client])

  // ── load dubbing-texts and files when open job changes ────────────────────
  useEffect(() => {
    if (!client || !selectedJobId) return
    setIsLoading(true)
    setLoadError('')

    // 1. Fetch dubbing texts for the selected job
    client
      .get<TextDocument>(`/api/jobs/${encodeURIComponent(selectedJobId)}/dubbing-texts`)
      .then((doc) => {
        setDocument(doc)
        setSegments(doc.segments)
        setRevision(doc.revision)
        setDirty(false)
        if (doc.segments.length > 0) setActiveSegmentId(doc.segments[0].segment_id)
      })
      .catch((err: unknown) => {
        setLoadError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => setIsLoading(false))

    // 2. Fetch specific video file associated with the selected job
    client
      .get<{ files?: JobFile[] }>(`/api/jobs/${encodeURIComponent(selectedJobId)}/files`)
      .then((res) => {
        const fileList = Array.isArray(res.files) ? res.files : []
        const videoFile = fileList.find(
          (f) =>
            f.kind === 'video' ||
            f.kind === 'output_video' ||
            f.kind === 'input_video' ||
            /\.(mp4|webm|mov|mkv)$/i.test(f.name),
        )
        if (videoFile) {
          const fileUrl =
            videoFile.url || `/api/jobs/${encodeURIComponent(selectedJobId)}/files/${encodeURIComponent(videoFile.id)}`
          setVideoSrc(fileUrl)
        } else {
          setVideoSrc('/video.mp4')
        }
      })
      .catch(() => {
        setVideoSrc('/video.mp4')
      })
  }, [client, selectedJobId])

  // ── edit translation ───────────────────────────────────────────────────────
  const handleTranslationChange = (id: string, value: string) => {
    setSegments((prev) => prev.map((s) => s.segment_id === id ? { ...s, translation: value } : s))
    setDirty(true)
  }

  // ── save all changes ───────────────────────────────────────────────────────
  const handleSave = async () => {
    if (!client || !selectedJobId || !document) return
    setStatusMessage('Saving changes…')
    try {
      const updated = await client.put<TextDocument>(
        `/api/jobs/${encodeURIComponent(selectedJobId)}/dubbing-texts`,
        { revision, segments: segments.map(({ audio: _a, ...s }) => s) },
      )
      setDocument(updated)
      setSegments(updated.segments)
      setRevision(updated.revision)
      setDirty(false)
      setStatusMessage('Saved successfully.')
    } catch (err) {
      setStatusMessage(err instanceof Error ? err.message : 'Save failed.')
    } finally {
      setTimeout(() => setStatusMessage(''), 3000)
    }
  }

  // ── regenerate audio for one segment ─────────────────────────────────────
  const handleRegenerateAudio = async (segmentId: string) => {
    const seg = segments.find((s) => s.segment_id === segmentId)
    if (!seg || !client || !selectedJobId) return
    setIsRegenerating(true)
    setStatusMessage('Regenerating audio…')
    try {
      const result = await client.post<{ revision: string; segment: Segment }>(
        `/api/jobs/${encodeURIComponent(selectedJobId)}/dubbing-texts/${encodeURIComponent(segmentId)}/regenerate`,
        { revision, synthesized_text: seg.synthesized_text || seg.translation },
      )
      setRevision(result.revision)
      setSegments((prev) =>
        prev.map((s) =>
          s.segment_id === segmentId
            ? { ...s, synthesized_text: result.segment.synthesized_text, audio: result.segment.audio }
            : s,
        ),
      )
      setStatusMessage('Audio ready.')
    } catch (err) {
      setStatusMessage(err instanceof Error ? err.message : 'Regeneration failed.')
    } finally {
      setIsRegenerating(false)
      setTimeout(() => setStatusMessage(''), 3000)
    }
  }

  // ── mute toggle per speaker track ─────────────────────────────────────────
  const toggleTrackMute = (trackId: string) =>
    setMutedTracks((prev) => ({ ...prev, [trackId]: !prev[trackId] }))

  // ── search filter ─────────────────────────────────────────────────────────
  const filtered = displaySegments.filter(
    (s) =>
      s.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.translation.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.speaker.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const currentActive = displaySegments.find((s) => s.segment_id === activeSegmentId) ?? displaySegments[0]

  // Find segment corresponding to current playback time or active selection
  const liveSubSegment = displaySegments.find((s) => currentTime >= s.start && currentTime <= s.end) ?? currentActive

  // ── video controls handlers ───────────────────────────────────────────────
  const togglePlayPause = () => {
    if (!videoRef.current) return
    if (isPlaying) {
      videoRef.current.pause()
      setIsPlaying(false)
    } else {
      void videoRef.current.play()
      setIsPlaying(true)
    }
  }

  const handleSelectSegment = (seg: Segment) => {
    setActiveSegmentId(seg.segment_id)
    if (videoRef.current) {
      videoRef.current.currentTime = seg.start
      setCurrentTime(seg.start)
    }
  }

  const handleScrubberClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const newTime = pct * duration
    if (videoRef.current) {
      videoRef.current.currentTime = newTime
      setCurrentTime(newTime)
    }
  }

  // ── timeline clip positions ────────────────────────────────────────────────
  const maxTime = displaySegments.length > 0 ? Math.max(...displaySegments.map((s) => s.end)) : 60
  const duration = Math.max(videoDuration || 0, maxTime, 60)

  const clipStyle = (s: Segment) => ({
    left: `${(s.start / duration) * 100}%`,
    width: `${Math.max(3, ((s.end - s.start) / duration) * 100)}%`,
  })

  return (
    <section className="studio-view-container">
      <h1 className="sr-only">HukFlow Studio</h1>

      {/* ── Main Split Workspace ─────────────────────────────────────────── */}
      <div className="studio-workspace">

        {/* LEFT PANE: Transcript ───────────────────────────────────────────── */}
        <div className="studio-left-pane">
          <div className="pane-header">
            <div className="pane-header-left">
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
              {dirty && (
                <button type="button" className="save-badge-btn" onClick={handleSave}>
                  <span className="material-symbols-outlined">save</span>
                  Save
                </button>
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
              <button className="icon-btn" title="Filter" type="button">
                <span className="material-symbols-outlined">filter_list</span>
              </button>
            </div>
          </div>

          {/* Status & Error Banners */}
          {statusMessage && (
            <div className="studio-status-banner" role="status">{statusMessage}</div>
          )}
          {loadError && (
            <div className="studio-error-banner" role="alert">{loadError}</div>
          )}

          <div className="transcript-scroll-area">
            {isLoading && (
              <div className="transcript-loading">
                <span className="material-symbols-outlined spin-anim">sync</span>
                <span>Loading dubbing texts…</span>
              </div>
            )}

            {filtered.map((seg, idx) => {
              const isActive = seg.segment_id === (currentActive?.segment_id ?? '')
              const hasAudio = seg.audio !== null
              const speakerSlot = SPEAKER_COLOURS[idx % 4]

              return (
                <div
                  key={seg.segment_id}
                  onClick={() => handleSelectSegment(seg)}
                  className={`transcript-card ${isActive ? 'active-card' : ''}`}
                  data-testid={`transcript-card-${seg.segment_id}`}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => e.key === 'Enter' && handleSelectSegment(seg)}
                >
                  <div className={`card-accent-line ${speakerSlot}`}></div>

                  <div className="card-header">
                    <div className="speaker-info">
                      <span className={`speaker-dot speaker-dot-${idx % 4}`}></span>
                      <span className="speaker-name">{seg.speaker}</span>
                      <span className="time-badge">
                        {formatTime(seg.start)} – {formatTime(seg.end)}
                      </span>
                    </div>
                    <div className="card-status-row">
                      {hasAudio ? (
                        <span className="material-symbols-outlined status-completed" title="Audio ready">check_circle</span>
                      ) : (
                        <span className="material-symbols-outlined status-pending" title="Audio missing">pending</span>
                      )}
                      {seg.audio && (
                        <a
                          href={seg.audio.url}
                          className="audio-play-link"
                          title={`Play ${seg.audio.name}`}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <span className="material-symbols-outlined">play_circle</span>
                        </a>
                      )}
                    </div>
                  </div>

                  <div className="card-grid">
                    <div className="source-col">
                      <p className="col-label">Source (EN)</p>
                      <p className={`source-text source-text-${speakerSlot.replace('speaker-', '')}`}>{seg.text}</p>
                    </div>

                    <div className="dubbed-col">
                      <p className="col-label">Dubbed (BE)</p>
                      {isActive ? (
                        <div className="editable-dubbed-wrapper">
                          <textarea
                            className="dubbed-textarea"
                            value={seg.translation}
                            onChange={(e) => handleTranslationChange(seg.segment_id, e.target.value)}
                            onClick={(e) => e.stopPropagation()}
                            rows={3}
                            aria-label={`Dubbed text ${seg.speaker}`}
                          />
                          <button
                            type="button"
                            className={`regenerate-btn ${isRegenerating ? 'spinning' : ''}`}
                            onClick={(e) => {
                              e.stopPropagation()
                              void handleRegenerateAudio(seg.segment_id)
                            }}
                            title="Regenerate Audio"
                            disabled={isRegenerating}
                          >
                            <span className="material-symbols-outlined">autorenew</span>
                          </button>
                        </div>
                      ) : (
                        <div className="dubbed-read-only">
                          {seg.translation || <em className="empty-translation">No translation</em>}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* RIGHT PANE: Video Player ─────────────────────────────────────────── */}
        <div className="studio-right-pane">
          <div className="player-top-bar">
            <span className="program-title">
              {document ? `Source: ${document.source}` : 'Program: Main_Edit_v2'}
            </span>
            <span className="video-specs">
              {selectedJobId ? `Job: ${selectedJobId}` : '1080p | 23.976 fps'}
            </span>
          </div>

          <div className="video-preview-wrapper">
            <div className="video-canvas">
              <video
                ref={videoRef}
                src={videoSrc}
                className="video-poster"
                onTimeUpdate={() => {
                  if (videoRef.current) setCurrentTime(videoRef.current.currentTime)
                }}
                onLoadedMetadata={() => {
                  if (videoRef.current && videoRef.current.duration) {
                    setVideoDuration(videoRef.current.duration)
                  }
                }}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onClick={togglePlayPause}
              />

              <div className="subtitle-overlay">
                <span>
                  {liveSubSegment ? liveSubSegment.translation || liveSubSegment.text : 'Select or play video segment'}
                </span>
              </div>
            </div>
          </div>

          {/* Transport controls */}
          <div className="player-controls-bar">
            <div className="timecode-display">
              <span>{formatTimecode(currentTime)}</span>
              <span>{formatTimecode(duration)}</span>
            </div>

            <div className="player-scrubber" onClick={handleScrubberClick}>
              <div className="scrubber-track">
                <div
                  className="scrubber-progress"
                  style={{
                    width: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%',
                  }}
                ></div>
                <div
                  className="scrubber-handle"
                  style={{
                    left: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%',
                  }}
                ></div>
              </div>
            </div>

            <div className="transport-buttons">
              <button
                type="button"
                className="transport-btn"
                title="Skip Previous"
                onClick={() => {
                  const idx = displaySegments.findIndex((s) => s.segment_id === currentActive?.segment_id)
                  if (idx > 0) handleSelectSegment(displaySegments[idx - 1])
                }}
              >
                <span className="material-symbols-outlined">skip_previous</span>
              </button>
              <button
                type="button"
                className="transport-btn"
                title="Fast Rewind -5s"
                onClick={() => {
                  if (videoRef.current) {
                    const newTime = Math.max(0, videoRef.current.currentTime - 5)
                    videoRef.current.currentTime = newTime
                    setCurrentTime(newTime)
                  }
                }}
              >
                <span className="material-symbols-outlined">fast_rewind</span>
              </button>
              <button
                type="button"
                className="play-pause-btn"
                title={isPlaying ? 'Pause' : 'Play'}
                onClick={togglePlayPause}
              >
                <span className="material-symbols-outlined">
                  {isPlaying ? 'pause' : 'play_arrow'}
                </span>
              </button>
              <button
                type="button"
                className="transport-btn"
                title="Fast Forward +5s"
                onClick={() => {
                  if (videoRef.current) {
                    const newTime = Math.min(duration, videoRef.current.currentTime + 5)
                    videoRef.current.currentTime = newTime
                    setCurrentTime(newTime)
                  }
                }}
              >
                <span className="material-symbols-outlined">fast_forward</span>
              </button>
              <button
                type="button"
                className="transport-btn"
                title="Skip Next"
                onClick={() => {
                  const idx = displaySegments.findIndex((s) => s.segment_id === currentActive?.segment_id)
                  if (idx < displaySegments.length - 1) handleSelectSegment(displaySegments[idx + 1])
                }}
              >
                <span className="material-symbols-outlined">skip_next</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── Docked Bottom Timeline ─────────────────────────────────────────── */}
      <div className="studio-bottom-timeline">

        {/* Timeline toolbar */}
        <div className="timeline-toolbar">
          <div className="toolbar-tools">
            {(['razor', 'sync', 'clone'] as const).map((tool) => (
              <button
                key={tool}
                type="button"
                className={`tool-btn ${activeTool === tool ? 'active' : ''}`}
                title={tool === 'razor' ? 'Razor Cut' : tool === 'sync' ? 'Sync Audio' : 'Voice Clone'}
                onClick={() => setActiveTool(activeTool === tool ? null : tool)}
              >
                <span className="material-symbols-outlined">
                  {tool === 'razor' ? 'content_cut' : tool === 'sync' ? 'sync_alt' : 'record_voice_over'}
                </span>
              </button>
            ))}
          </div>

          <div className="toolbar-zoom">
            <span className="material-symbols-outlined zoom-icon">zoom_out</span>
            <input
              type="range" min="10" max="100" value={zoomLevel}
              onChange={(e) => setZoomLevel(Number(e.target.value))}
              className="zoom-slider"
              aria-label="Zoom timeline"
            />
            <span className="material-symbols-outlined zoom-icon">zoom_in</span>
          </div>
        </div>

        {/* Timeline body */}
        <div className="timeline-body">
          {/* Track header column */}
          <div className="track-headers-column">
            <div className="ruler-corner"></div>
            {displaySpeakers.map((speaker, idx) => (
              <div key={speaker} className={`track-header border-speaker-${idx % 4}`}>
                <span className="track-name" title={speaker}>
                  V{idx + 1}: {speaker.replace('SPEAKER_', 'Spk ')}
                </span>
                <button
                  type="button"
                  className="mute-btn"
                  onClick={() => toggleTrackMute(speaker)}
                  title="Toggle Mute"
                >
                  <span className="material-symbols-outlined">
                    {mutedTracks[speaker] ? 'volume_off' : 'volume_up'}
                  </span>
                </button>
              </div>
            ))}
            {/* Music track */}
            <div className="track-header border-music">
              <span className="track-name">A1: Music</span>
              <button type="button" className="mute-btn" onClick={() => toggleTrackMute('a1')} title="Toggle Mute">
                <span className="material-symbols-outlined">{mutedTracks.a1 ? 'volume_off' : 'volume_up'}</span>
              </button>
            </div>
          </div>

          {/* Timeline canvas */}
          <div className="timeline-tracks-canvas">
            {/* Time ruler */}
            <div className="time-ruler">
              {Array.from({ length: 5 }, (_, i) => (
                <span key={i}>{formatTime((duration / 4) * i)}</span>
              ))}
            </div>

            {/* Playhead at video currentTime */}
            <div
              className="timeline-playhead"
              style={{
                left: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%',
              }}
            >
              <div className="playhead-head"></div>
            </div>

            {/* Speaker track rows from real segments */}
            {displaySpeakers.map((speaker, idx) => {
              const speakerSegs = displaySegments.filter((s) => s.speaker === speaker)
              return (
                <div key={speaker} className="track-row">
                  {speakerSegs.map((seg) => {
                    const isActiveClip = seg.segment_id === (currentActive?.segment_id ?? '')
                    const clipClass = isActiveClip
                      ? `clip-block primary-clip${seg.audio === null ? ' ai-shimmer' : ''}`
                      : `clip-block speaker-clip-${idx % 4}`
                    return (
                      <div
                        key={seg.segment_id}
                        className={clipClass}
                        style={clipStyle(seg)}
                        title={seg.translation || seg.text}
                        onClick={() => handleSelectSegment(seg)}
                        role="button"
                        tabIndex={-1}
                      >
                        <svg className="waveform-svg" viewBox="0 0 100 20" preserveAspectRatio="none">
                          <path
                            d="M0,10 L5,4 L10,16 L15,6 L20,14 L25,8 L30,12 L35,4 L40,16 L45,10 L50,7 L55,13 L60,4 L65,16 L70,9 L75,11 L80,6 L85,14 L90,8 L95,12 L100,10"
                            fill="none" stroke="currentColor" strokeWidth="1.5"
                          />
                        </svg>
                        {isActiveClip && (
                          <span className="clip-subtitle">
                            {seg.translation || seg.text}
                          </span>
                        )}
                      </div>
                    )
                  })}
                </div>
              )
            })}

            {/* Music track row */}
            <div className="track-row">
              <div className="clip-block music-clip" style={{ left: '0%', width: '98%' }}>
                <svg className="waveform-svg" viewBox="0 0 200 20" preserveAspectRatio="none">
                  <path d="M0,10 Q10,5 20,10 T40,10 T60,10 T80,10 T100,10 T120,10 T140,10 T160,10 T180,10 T200,10" fill="none" stroke="currentColor" strokeWidth="1" />
                </svg>
                <span className="clip-label">Background.wav</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
