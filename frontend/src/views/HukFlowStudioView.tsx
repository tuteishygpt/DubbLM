import { useEffect, useRef, useState } from 'react'
import type { ApiClient } from '../api/types'

// ── API types ─────────────────────────────────────────────────────────────
interface Job { id: string; status: string }

interface ProjectSummary {
  name: string
  relative_path: string
  has_video: boolean
  has_subtitles: boolean
  has_artifacts: boolean
  has_transcription: boolean
  segment_count: number
  video_files: string[]
  audio_files: string[]
  job_id: string | null
}

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
  const rounded = Math.round(seconds)
  const m = Math.floor(rounded / 60)
  const s = Math.floor(rounded % 60)
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
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [selectedProjectName, setSelectedProjectName] = useState<string>('')
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
  const [videoDuration, setVideoDuration] = useState(0)
  const [musicFiles, setMusicFiles] = useState<JobFile[]>([])
  const [backgroundDurations, setBackgroundDurations] = useState<Record<string, number>>({})

  // ── studio tools state ───────────────────────────────────────────────────
  const [activeTool, setActiveTool] = useState<'razor' | 'sync' | 'clone' | null>(null)
  const [zoomLevel, setZoomLevel] = useState(50)
  const [mutedTracks, setMutedTracks] = useState<Record<string, boolean>>({ a1: true })
  const [searchQuery, setSearchQuery] = useState('')
  const [isRegenerating, setIsRegenerating] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({})
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  const toggleSource = (segmentId: string) => {
    setExpandedSources((prev) => ({
      ...prev,
      [segmentId]: !prev[segmentId],
    }))
  }

  // Fallback demo segments ONLY when running in standalone preview mode without API client
  const isStandaloneDemo = !client && !selectedJobId
  const displaySegments = isStandaloneDemo && segments.length === 0 ? [
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
  ] : segments

  const displaySpeakers = [...new Set(displaySegments.map((s) => s.speaker))]

  // ── load jobs and ready projects ──────────────────────────────────────────
  useEffect(() => {
    if (!client) return
    client.get<{ jobs: Job[] }>('/api/jobs')
      .then((res) => {
        const list = Array.isArray(res.jobs) ? res.jobs : []
        setJobs(list)
        if (list.length > 0) setSelectedJobId((prev) => prev || list[0].id)
      })
      .catch(() => {/* demo mode */})

    client.get<{ projects?: ProjectSummary[] }>('/api/projects')
      .then((res) => {
        const list = Array.isArray(res?.projects) ? res.projects : []
        setProjects(list)
      })
      .catch(() => {/* ignore if endpoint not present */})
  }, [client])

  // ── load dubbing-texts and files when open job changes ────────────────────
  useEffect(() => {
    if (!client || !selectedJobId) return
    setIsLoading(true)
    setLoadError('')
    setDocument(null)
    setSegments([])
    setActiveSegmentId('')

    // 1. Fetch dubbing texts for the selected job
    client
      .get<TextDocument>(`/api/jobs/${encodeURIComponent(selectedJobId)}/dubbing-texts`)
      .then((doc) => {
        const loadedSegments = Array.isArray(doc?.segments) ? doc.segments : []
        setDocument(doc)
        setSegments(loadedSegments)
        setRevision(doc?.revision || '')
        setDirty(false)
        if (loadedSegments.length > 0) setActiveSegmentId(loadedSegments[0].segment_id)
      })
      .catch((err: unknown) => {
        setLoadError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => setIsLoading(false))

    // 2. Fetch specific video and audio files associated with the selected job
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

        const audioFiles = fileList.filter(
          (f) =>
            f.kind === 'music' ||
            f.kind === 'background_audio' ||
            f.kind === 'audio' ||
            /background|music/i.test(f.name) ||
            /\.(mp3|wav|m4a|aac|flac)$/i.test(f.name),
        )
        setMusicFiles(audioFiles)
      })
      .catch(() => {
        setVideoSrc('/video.mp4')
        setMusicFiles([])
      })
  }, [client, selectedJobId])

  // ── fetch duration for background audio files ─────────────────────────────
  useEffect(() => {
    if (musicFiles.length === 0) {
      setBackgroundDurations({})
      return
    }

    musicFiles.forEach((file) => {
      const url =
        file.url ||
        (selectedJobId ? `/api/jobs/${encodeURIComponent(selectedJobId)}/files/${encodeURIComponent(file.id)}` : '')
      if (!url) return
      const tempAudio = new Audio(url)
      const onLoaded = () => {
        if (Number.isFinite(tempAudio.duration) && tempAudio.duration > 0) {
          setBackgroundDurations((prev) => ({ ...prev, [file.id]: tempAudio.duration }))
        }
      }
      tempAudio.addEventListener('loadedmetadata', onLoaded)
    })
  }, [musicFiles, selectedJobId])

  // ── edit translation ───────────────────────────────────────────────────────
  const handleTranslationChange = (id: string, value: string) => {
    setSegments((prev) => prev.map((s) => s.segment_id === id ? { ...s, translation: value } : s))
    setDirty(true)
  }

  // ── open ready project from prj/ ───────────────────────────────────────────
  const handleOpenProject = async (projectName: string) => {
    if (!client || !projectName) return
    setSelectedProjectName(projectName)
    setIsLoading(true)
    setStatusMessage(`Opening project ${projectName}…`)
    try {
      const res = await client.post<{ project_name: string; job: Job }>(
        `/api/projects/${encodeURIComponent(projectName)}/open`,
      )
      if (res?.job?.id) {
        setJobs((prev) => {
          const exists = prev.some((j) => j.id === res.job.id)
          return exists ? prev : [res.job, ...prev]
        })
        setSelectedJobId(res.job.id)
        setStatusMessage(`Opened project ${projectName}`)
      }
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsLoading(false)
      setTimeout(() => setStatusMessage(''), 3000)
    }
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

  // ── search filter & active segment ───────────────────────────────────────
  const filtered = displaySegments.filter(
    (s) =>
      s.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.translation.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.speaker.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const visibleSegments = searchQuery.trim() ? filtered : displaySegments

  // Find segment corresponding to current playback time (if currentTime falls within segment bounds)
  const liveSubSegment = visibleSegments.find((s) => currentTime >= s.start && currentTime <= s.end)

  // Current active segment: live segment during playback/scrubbing, otherwise selected segment, fallback to first visible segment
  const currentActive =
    liveSubSegment ??
    visibleSegments.find((s) => s.segment_id === activeSegmentId) ??
    (visibleSegments.length > 0 ? visibleSegments[0] : null)

  // ── timeline clip positions (mastered to video duration when present) ───────────
  const maxTime = displaySegments.length > 0 ? Math.max(...displaySegments.map((s) => s.end)) : 0
  const rawDuration = videoDuration > 0 ? videoDuration : maxTime > 0 ? maxTime : 60
  const duration = Math.max(0.1, rawDuration)

  const clipStyle = (s: Segment) => {
    const start = Math.min(Math.max(0, s.start), duration)
    const end = Math.min(Math.max(start, s.end), duration)
    const clipLen = end - start
    return {
      left: `${(start / duration) * 100}%`,
      width: `${Math.max(0.5, (clipLen / duration) * 100)}%`,
    }
  }

  const seekToTime = (targetTime: number) => {
    const newTime = Math.max(0, Math.min(duration, targetTime))
    setCurrentTime(newTime)
    if (videoRef.current) {
      const vidTime = videoDuration > 0 ? Math.min(newTime, videoDuration) : newTime
      videoRef.current.currentTime = vidTime
    }
  }

  // ── video controls handlers ───────────────────────────────────────────────
  const togglePlayPause = () => {
    if (isPlaying) {
      if (videoRef.current) videoRef.current.pause()
      setIsPlaying(false)
    } else {
      const startFrom = currentTime >= duration - 0.1 ? 0 : currentTime
      setCurrentTime(startFrom)
      if (videoRef.current) {
        const vidTime = videoDuration > 0 ? Math.min(startFrom, videoDuration) : startFrom
        videoRef.current.currentTime = vidTime
        if (startFrom < (videoDuration || duration)) {
          void videoRef.current.play().catch(() => {/* ignore autoplay restrictions */})
        }
      }
      setIsPlaying(true)
    }
  }

  const handleSelectSegment = (seg: Segment) => {
    setActiveSegmentId(seg.segment_id)
    seekToTime(seg.start)
  }

  // Auto-scroll active card into view when currentActive changes
  useEffect(() => {
    if (!currentActive?.segment_id || !scrollAreaRef.current) return
    const el = scrollAreaRef.current.querySelector<HTMLElement>(`[data-testid="transcript-card-${currentActive.segment_id}"]`)
    if (el && typeof el.scrollIntoView === 'function') {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [currentActive?.segment_id])

  const handleScrubberClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    seekToTime(pct * duration)
  }

  // ── continuous playback timer ──────────────────────────────────────────────
  useEffect(() => {
    if (!isPlaying) return

    let animationFrameId: number
    let lastTime = performance.now()

    const tick = (now: number) => {
      const dt = (now - lastTime) / 1000
      lastTime = now

      setCurrentTime((prev) => {
        let next: number
        if (
          videoRef.current &&
          videoDuration > 0 &&
          videoRef.current.currentTime < videoDuration - 0.1 &&
          !videoRef.current.paused
        ) {
          next = videoRef.current.currentTime
        } else {
          next = prev + dt
        }

        if (next >= duration) {
          setIsPlaying(false)
          if (videoRef.current) videoRef.current.pause()
          return duration
        }
        return next
      })

      animationFrameId = requestAnimationFrame(tick)
    }

    animationFrameId = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animationFrameId)
  }, [isPlaying, videoDuration, duration])

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
                      📁 {p.name} {p.segment_count > 0 ? `(${p.segment_count} segments)` : ''}
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
              visibleSegments.map((seg, idx) => {
                const isActive = seg.segment_id === (currentActive?.segment_id ?? '')
                const hasAudio = seg.audio !== null
                const speakerSlot = SPEAKER_COLOURS[idx % 4]
                const isSourceExpanded = Boolean(expandedSources[seg.segment_id])
                // Calculate dynamic row count based on text length
                const dynamicRows = Math.max(3, Math.min(8, Math.ceil((seg.translation.length || 1) / 45)))

                return (
                  <div
                    key={seg.segment_id}
                    id={`transcript-card-${seg.segment_id}`}
                    onClick={() => handleSelectSegment(seg)}
                    className={`transcript-card ${isActive ? 'active-card' : 'inactive-card'}`}
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
                        {visibleSegments.length > 1 && (
                          <span className="segment-stepper-label">
                            ({idx + 1}/{visibleSegments.length})
                          </span>
                        )}
                      </div>

                      <div className="card-status-row">
                        {/* Source text toggle icon button */}
                        <button
                          type="button"
                          className={`source-toggle-btn ${isSourceExpanded ? 'active' : ''}`}
                          onClick={(e) => {
                            e.stopPropagation()
                            toggleSource(seg.segment_id)
                          }}
                          title={`Source: ${seg.text}`}
                          aria-label={`View source text for ${seg.speaker}`}
                        >
                          <span className="material-symbols-outlined source-icon">menu_book</span>
                          <span className="source-toggle-label">EN</span>
                        </button>

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

                    {/* Expanded Source Text Drawer */}
                    {isSourceExpanded && (
                      <div className="source-drawer" onClick={(e) => e.stopPropagation()}>
                        <div className="source-drawer-header">
                          <span className="col-label">Source ({seg.speaker})</span>
                          <button
                            type="button"
                            className="icon-btn close-source-btn"
                            onClick={() => toggleSource(seg.segment_id)}
                            title="Hide source text"
                          >
                            <span className="material-symbols-outlined">close</span>
                          </button>
                        </div>
                        <p className={`source-text source-text-${speakerSlot.replace('speaker-', '')}`}>
                          {seg.text}
                        </p>
                      </div>
                    )}

                    {/* Dubbed Editor — full width with dynamic height */}
                    <div className="active-dubbed-area">
                      <div className="dubbed-area-header">
                        <span className="col-label">Dubbed (BE)</span>
                        <span className="text-count-badge">{seg.translation.length} chars</span>
                      </div>
                      <div className="editable-dubbed-wrapper">
                        <textarea
                          className="dubbed-textarea"
                          value={seg.translation}
                          onChange={(e) => handleTranslationChange(seg.segment_id, e.target.value)}
                          onClick={(e) => {
                            e.stopPropagation()
                            handleSelectSegment(seg)
                          }}
                          onFocus={() => handleSelectSegment(seg)}
                          rows={dynamicRows}
                          aria-label={`Dubbed text ${seg.speaker}`}
                          placeholder="Dubbed text in Belarusian..."
                        />
                        {isActive && (
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
          <div className="video-preview-wrapper">
            <div className="video-canvas">
              <video
                ref={videoRef}
                src={videoSrc}
                className="video-poster"
                onTimeUpdate={() => {
                  if (videoRef.current) setCurrentTime(videoRef.current.currentTime)
                }}
                onSeeking={() => {
                  if (videoRef.current) setCurrentTime(videoRef.current.currentTime)
                }}
                onSeeked={() => {
                  if (videoRef.current) setCurrentTime(videoRef.current.currentTime)
                }}
                onLoadedMetadata={() => {
                  if (videoRef.current && Number.isFinite(videoRef.current.duration) && videoRef.current.duration > 0) {
                    setVideoDuration(videoRef.current.duration)
                  }
                }}
                onPause={() => {
                  if (videoDuration > 0 && currentTime < videoDuration - 0.3) {
                    setIsPlaying(false)
                  } else if (videoDuration <= 0 && currentTime < duration - 0.3) {
                    setIsPlaying(false)
                  }
                }}
                onClick={togglePlayPause}
              />

              <div className="subtitle-overlay">
                {liveSubSegment && (
                  <span>
                    {liveSubSegment.translation || liveSubSegment.text}
                  </span>
                )}
              </div>

              {/* Overlay transport controls */}
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
          <div
            className="timeline-tracks-canvas"
            onClick={(e) => {
              if ((e.target as HTMLElement).closest('.clip-block')) return
              const rect = e.currentTarget.getBoundingClientRect()
              const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
              seekToTime(pct * duration)
            }}
          >
            {/* Time ruler */}
            <div className="time-ruler">
              {Array.from({ length: 5 }, (_, i) => (
                <span
                  key={i}
                  style={{
                    position: 'absolute',
                    left: `${(i / 4) * 100}%`,
                    transform: i === 0 ? 'none' : i === 4 ? 'translateX(-100%)' : 'translateX(-50%)',
                  }}
                >
                  {formatTime((duration / 4) * i)}
                </span>
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
              {musicFiles.length > 0 ? (
                musicFiles.map((m) => {
                  const audioDur = backgroundDurations[m.id]
                  const widthPct = audioDur && duration > 0 ? Math.min(100, (audioDur / duration) * 100) : 100
                  return (
                    <div
                      key={m.id}
                      className="clip-block music-clip"
                      style={{ left: '0%', width: `${widthPct}%` }}
                      title={`${m.name}${audioDur ? ` (${formatTime(audioDur)})` : ''}`}
                    >
                      <svg className="waveform-svg" viewBox="0 0 200 20" preserveAspectRatio="none">
                        <path d="M0,10 Q10,5 20,10 T40,10 T60,10 T80,10 T100,10 T120,10 T140,10 T160,10 T180,10 T200,10" fill="none" stroke="currentColor" strokeWidth="1" />
                      </svg>
                      <span className="clip-label">{m.name}</span>
                    </div>
                  )
                })
              ) : (
                <div className="clip-block music-clip" style={{ left: '0%', width: '100%' }}>
                  <svg className="waveform-svg" viewBox="0 0 200 20" preserveAspectRatio="none">
                    <path d="M0,10 Q10,5 20,10 T40,10 T60,10 T80,10 T100,10 T120,10 T140,10 T160,10 T180,10 T200,10" fill="none" stroke="currentColor" strokeWidth="1" />
                  </svg>
                  <span className="clip-label">Background.wav</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
