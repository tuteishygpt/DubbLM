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
    <div className="bg-background text-on-background font-body-md text-body-md overflow-hidden selection:bg-primary-container selection:text-on-primary-container min-h-screen flex flex-col relative">
      <h1 className="sr-only">HukFlow Studio</h1>

      {/* TopNavBar */}
      <nav className="flex justify-between items-center h-12 px-gutter w-full z-50 bg-background dark:bg-background bg-surface-container-low border-b border-outline-variant fixed top-0 left-0 right-0">
        <div className="flex items-center gap-stack-md">
          <button className="text-on-surface-variant hover:bg-surface-container-highest transition-colors duration-150 rounded p-1 flex items-center justify-center" title="Back to Home">
            <span className="material-symbols-outlined">arrow_back</span>
          </button>
          <span className="font-headline-md text-headline-md font-bold text-primary dark:text-primary tracking-tighter">HukFlow</span>
          <div className="h-6 w-px bg-outline-variant mx-1"></div>
          <div className="flex items-center gap-2">
            <img
              alt="Project Thumbnail"
              className="w-8 h-8 rounded object-cover"
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuBNYiL9TzjyYVVg9MeFaymgODiw6PGlfTptbTGqUpcDY8tHKwrT32x5f3qco7G30Vh-mOYFiZpvyQ2SeBaKbNY9mA0cfonYXgz2KAUf5R41LFdiMH0wkCGfPTCy81cTjnd4-pJNJh9i1dTmWQuC7Ziet4AlSEbJsbzNEJcQf7Xtgsy2DR5f-n3FviPp6z6ORJW6b_cgSelo7o3GgQY99uCFzOQP8uyb9av6-YsZp2qwxEK5Xrh3Y3m0mQ"
            />
            <div className="flex flex-col justify-center">
              <span className="font-label-md text-label-md text-primary font-bold leading-tight">
                {selectedJobId ? `Job: ${selectedJobId}` : 'Project Alpha'}
              </span>
              <span className="font-label-sm text-[10px] text-on-surface-variant leading-tight">Localizing: EN to BE</span>
            </div>
          </div>
          {jobs.length > 0 && (
            <select
              aria-label="Job selector"
              className="bg-surface-container-highest text-on-surface-variant border border-outline-variant rounded px-2 py-0.5 text-xs ml-2"
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
            <button type="button" className="bg-primary-container text-on-primary-container px-2 py-0.5 rounded text-xs flex items-center gap-1 font-bold hover:opacity-90" onClick={handleSave}>
              <span className="material-symbols-outlined text-[14px]">save</span>
              Save
            </button>
          )}
        </div>

        <div className="flex items-center gap-stack-md">
          <button className="text-on-surface-variant hover:bg-surface-container-highest transition-colors duration-150 rounded p-1" title="Sync Status">
            <span className="material-symbols-outlined">cloud_done</span>
          </button>
          <button className="text-on-surface-variant hover:bg-surface-container-highest transition-colors duration-150 rounded p-1" title="Settings">
            <span className="material-symbols-outlined">settings</span>
          </button>
          <button className="text-on-surface-variant hover:bg-surface-container-highest transition-colors duration-150 rounded px-3 py-1 font-label-md text-label-md">Share</button>
          <button className="bg-primary text-on-primary hover:bg-primary-fixed transition-colors duration-150 rounded px-4 py-1 font-label-md text-label-md font-bold">Export</button>
          <img
            alt="User profile"
            className="w-8 h-8 rounded-full ml-stack-sm object-cover border border-outline-variant"
            src="https://lh3.googleusercontent.com/aida-public/AB6AXuDJ8qlfIxbrXVkWD4yhJAd8e2NA4CR2DyzfH4SvtY8CRmx5GhpASNvrrf9Lq2Sxazp6z8x6Hbm681tCoPTEDKG2CZnFv0U5CFd8pnycAsFwmuQx0DBOWQETwxYVqF0LxN3q7GVbqslCgS4KPK3xVK4G4RmG5oA85IqDMFsS32C8Kp2j4FCXaEyK1yhLcY0hJ-oJdc5DEDZVBKcBt2LOpL_Xj-RHMPxW_Jv_3K0KseO1z4CAsV2pkOxf2Q"
          />
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="w-full mt-12 h-[calc(100vh-48px-240px)] flex bg-surface">
        {/* Left Pane: Transcript/Speaker Cards */}
        <div className="w-1/2 h-full border-r border-outline-variant flex flex-col">
          <div className="p-stack-md border-b border-outline-variant bg-surface-container-lowest flex justify-between items-center">
            <h3 className="font-headline-sm text-headline-sm text-on-surface">Localization Transcript</h3>
            <div className="flex items-center gap-2">
              <div className="relative flex items-center">
                <span className="material-symbols-outlined absolute left-2 text-[16px] text-on-surface-variant pointer-events-none">search</span>
                <input
                  type="text"
                  placeholder="Search transcript..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-surface-dim border border-outline-variant rounded pl-7 pr-2 py-1 text-xs text-on-surface focus:outline-none focus:border-primary w-36"
                />
              </div>
              <button className="p-1 rounded text-on-surface-variant hover:text-primary transition-colors" title="Filter">
                <span className="material-symbols-outlined">filter_list</span>
              </button>
            </div>
          </div>

          {/* Status & Error Banners */}
          {statusMessage && (
            <div className="bg-primary-container/20 border-b border-primary/30 text-primary text-xs px-4 py-1.5 font-medium">{statusMessage}</div>
          )}
          {loadError && (
            <div className="bg-error-container/20 border-b border-error/30 text-error text-xs px-4 py-1.5 font-medium">{loadError}</div>
          )}

          <div className="flex-1 overflow-y-auto p-stack-md space-y-stack-md bg-surface-dim px-gutter">
            {isLoading && (
              <div className="flex items-center justify-center p-8 gap-2 text-on-surface-variant">
                <span className="material-symbols-outlined animate-spin">sync</span>
                <span>Loading dubbing texts…</span>
              </div>
            )}

            {filtered.map((seg, idx) => {
              const isActive = seg.segment_id === (currentActive?.segment_id ?? '')
              const hasAudio = seg.audio !== null
              const isSpeaker1 = idx % 2 === 0

              return (
                <div
                  key={seg.segment_id}
                  onClick={() => handleSelectSegment(seg)}
                  className={`rounded-lg border relative transition-colors p-2 w-full cursor-pointer ${
                    isActive
                      ? 'bg-surface-container border-primary/50 shadow-[inset_0_0_10px_rgba(208,188,255,0.05)]'
                      : 'bg-surface-container border-outline-variant hover:border-secondary/50'
                  }`}
                  data-testid={`transcript-card-${seg.segment_id}`}
                >
                  <div
                    className={`absolute -left-[1px] top-4 bottom-4 w-[2px] rounded-r ${
                      isSpeaker1 ? 'bg-secondary' : 'bg-primary'
                    }`}
                  ></div>

                  <div className="flex justify-between items-start mb-stack-sm">
                    <div className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${isSpeaker1 ? 'bg-secondary' : 'bg-tertiary-container'}`}></span>
                      <span className="font-label-md text-label-md text-on-surface-variant">{seg.speaker}</span>
                      <span className="bg-surface-container-highest text-on-surface-variant px-2 py-0.5 rounded font-mono-ui text-mono-ui">
                        {formatTime(seg.start)} - {formatTime(seg.end)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {hasAudio ? (
                        <span className="material-symbols-outlined text-secondary text-[16px]" title="Audio ready">check_circle</span>
                      ) : (
                        <span className="material-symbols-outlined text-outline text-[16px]" title="Audio pending">pending</span>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-[1fr_2fr] gap-stack-md gap-gutter">
                    <div>
                      <p className="font-label-sm text-label-sm text-outline mb-1 uppercase tracking-wider">Source (EN)</p>
                      <p className="font-body-md text-body-md text-secondary">
                        <span className="text-headline-sm font-headline-sm">{seg.text}</span>
                      </p>
                    </div>
                    <div className="relative">
                      <p className="font-label-sm text-label-sm text-outline mb-1 uppercase tracking-wider">Dubbed (BE)</p>
                      {isActive ? (
                        <div className="relative">
                          <textarea
                            value={seg.translation}
                            onChange={(e) => handleTranslationChange(seg.segment_id, e.target.value)}
                            onClick={(e) => e.stopPropagation()}
                            className="w-full bg-surface-dim border border-primary rounded p-2 font-body-md text-body-md text-primary focus:border-primary focus:ring-0 resize-none text-headline-sm font-headline-sm h-32"
                          />
                          <button
                            type="button"
                            className={`absolute bottom-2 right-2 text-on-surface-variant hover:text-primary bg-surface-container-highest p-1 rounded transition-colors ${
                              isRegenerating ? 'animate-spin' : ''
                            }`}
                            onClick={(e) => {
                              e.stopPropagation()
                              void handleRegenerateAudio(seg.segment_id)
                            }}
                            title="Regenerate Audio"
                            disabled={isRegenerating}
                          >
                            <span className="material-symbols-outlined text-[16px]">autorenew</span>
                          </button>
                        </div>
                      ) : (
                        <div className="w-full bg-surface-dim border border-outline-variant rounded p-2 font-body-md text-body-md text-on-surface text-headline-sm font-headline-sm">
                          {seg.translation || <em className="text-outline text-xs">No translation</em>}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Right Pane: Video Player */}
        <div className="w-1/2 h-full bg-surface-container-lowest flex flex-col">
          <div className="p-stack-sm flex justify-between items-center bg-surface-container-low border-b border-outline-variant">
            <span className="font-label-md text-label-md text-on-surface">
              {document ? `Source: ${document.source}` : 'Program: Main_Edit_v2'}
            </span>
            <span className="font-mono-ui text-mono-ui text-primary">
              {selectedJobId ? `Job: ${selectedJobId}` : '1080p | 23.976 fps'}
            </span>
          </div>

          <div className="flex-1 p-stack-lg flex items-center justify-center relative">
            {/* HTML5 Video Element for current open job */}
            <div className="w-full aspect-video bg-black rounded-lg overflow-hidden border border-outline-variant relative flex items-center justify-center">
              <video
                ref={videoRef}
                src={videoSrc}
                className="w-full h-full object-contain"
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

              {/* Subtitles Overlay */}
              <div className="absolute bottom-6 left-0 right-0 text-center px-4 pointer-events-none">
                <span className="bg-black/80 text-white font-headline-sm text-headline-sm px-4 py-1.5 rounded backdrop-blur-md shadow-xl inline-block max-w-[90%] transition-all duration-200 border border-white/10">
                  {liveSubSegment ? liveSubSegment.translation || liveSubSegment.text : 'Select or play video segment'}
                </span>
              </div>
            </div>
          </div>

          {/* Player Controls */}
          <div className="p-stack-md bg-surface-container-low border-t border-outline-variant flex flex-col gap-stack-sm">
            <div className="flex justify-between items-center font-mono-ui text-mono-ui text-on-surface-variant">
              <span>{formatTimecode(currentTime)}</span>
              <span>{formatTimecode(duration)}</span>
            </div>

            <div className="h-1.5 bg-surface-container-highest rounded-full w-full relative cursor-pointer" onClick={handleScrubberClick}>
              <div
                className="absolute top-0 left-0 h-full bg-primary rounded-full transition-all duration-75"
                style={{
                  width: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%',
                }}
              ></div>
              <div
                className="absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 bg-primary rounded-full shadow-[0_0_8px_rgba(208,188,255,0.8)] transition-all duration-75 -ml-1.5"
                style={{
                  left: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%',
                }}
              ></div>
            </div>

            <div className="flex justify-center items-center gap-stack-lg mt-2">
              <button
                type="button"
                className="text-on-surface-variant hover:text-on-surface transition-colors"
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
                className="text-on-surface-variant hover:text-on-surface transition-colors"
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
                className="w-10 h-10 rounded-full bg-primary text-on-primary flex items-center justify-center hover:bg-primary-fixed transition-colors shadow-lg"
                onClick={togglePlayPause}
                title={isPlaying ? 'Pause' : 'Play'}
              >
                <span className="material-symbols-outlined" style={{ fontVariationSettings: '"FILL" 1' }}>
                  {isPlaying ? 'pause' : 'play_arrow'}
                </span>
              </button>
              <button
                type="button"
                className="text-on-surface-variant hover:text-on-surface transition-colors"
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
                className="text-on-surface-variant hover:text-on-surface transition-colors"
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
      </main>

      {/* BottomNavBar (Timeline) */}
      <div className="fixed bottom-0 right-0 left-0 z-30 h-[240px] bg-surface-container dark:bg-surface-container border-t border-outline-variant flex flex-col">
        {/* Timeline Toolbar */}
        <div className="flex items-center justify-between px-stack-md h-10 border-b border-surface-container-highest bg-surface-container-low">
          <div className="flex items-center gap-stack-lg">
            <div className="flex bg-surface-container-highest rounded p-0.5 gap-1">
              {(['razor', 'sync', 'clone'] as const).map((tool) => (
                <button
                  key={tool}
                  type="button"
                  onClick={() => setActiveTool(activeTool === tool ? null : tool)}
                  className={`p-1 rounded transition-colors ${
                    activeTool === tool ? 'text-primary bg-surface-container' : 'text-outline hover:text-primary hover:bg-surface-container-highest'
                  }`}
                  title={tool === 'razor' ? 'Razor' : tool === 'sync' ? 'Sync' : 'Voice Clone'}
                >
                  <span className="material-symbols-outlined text-[18px]">
                    {tool === 'razor' ? 'content_cut' : tool === 'sync' ? 'sync_alt' : 'record_voice_over'}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="h-4 w-px bg-outline-variant"></div>
            <div className="flex items-center gap-2 text-outline">
              <span className="material-symbols-outlined text-[16px]">zoom_out</span>
              <input
                type="range"
                min="10"
                max="100"
                value={zoomLevel}
                onChange={(e) => setZoomLevel(Number(e.target.value))}
                className="w-24 h-1 bg-surface-container-highest rounded-full accent-primary cursor-pointer"
                aria-label="Zoom timeline"
              />
              <span className="material-symbols-outlined text-[16px]">zoom_in</span>
            </div>
          </div>
        </div>

        {/* Timeline Tracks Area */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden relative flex">
          {/* Track Headers (Left fixed column) */}
          <div className="w-32 flex-shrink-0 bg-surface-container-low border-r border-surface-container-highest sticky left-0 z-10 flex flex-col gap-[2px] p-2">
            {/* Time Ruler Header */}
            <div className="h-6 bg-surface-container-lowest border-b border-surface-container-highest"></div>
            
            {displaySpeakers.map((speaker, idx) => (
              <div
                key={speaker}
                className={`h-16 bg-surface-container flex flex-col justify-center px-2 border-l-2 relative group hover:bg-surface-container-highest transition-colors ${
                  idx % 2 === 0 ? 'border-secondary' : 'border-tertiary-container'
                }`}
              >
                <div className="flex justify-between items-center w-full">
                  <span className="font-label-sm text-label-sm text-on-surface">
                    V{idx + 1}: {speaker.replace('SPEAKER_', 'Spk ')}
                  </span>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      type="button"
                      onClick={() => toggleTrackMute(speaker)}
                      className="text-outline hover:text-on-surface cursor-pointer"
                    >
                      <span className="material-symbols-outlined text-[14px]">
                        {mutedTracks[speaker] ? 'volume_off' : 'volume_up'}
                      </span>
                    </button>
                  </div>
                </div>
              </div>
            ))}

            <div className="h-16 bg-surface-container flex flex-col justify-center px-2 border-l-2 border-outline-variant relative group hover:bg-surface-container-highest transition-colors">
              <div className="flex justify-between items-center w-full">
                <span className="font-label-sm text-label-sm text-outline">A1: Music</span>
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    type="button"
                    onClick={() => toggleTrackMute('a1')}
                    className="text-outline hover:text-on-surface cursor-pointer"
                  >
                    <span className="material-symbols-outlined text-[14px]">
                      {mutedTracks.a1 ? 'volume_off' : 'volume_up'}
                    </span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Timeline Grid/Content */}
          <div className="flex-1 relative min-w-[800px] bg-[#1a1a1a] p-2">
            {/* Time Ruler */}
            <div className="h-6 bg-surface-container-lowest border-b border-surface-container-highest sticky top-0 z-0 flex items-end px-4 font-mono-ui text-[10px] text-outline-variant select-none">
              <div className="flex-1 flex justify-between">
                <span>00:00:00</span>
                <span>00:00:10</span>
                <span>00:00:20</span>
                <span>00:00:30</span>
                <span>00:00:40</span>
              </div>
            </div>

            {/* Playhead synchronized with HTML5 video player */}
            <div
              className="absolute top-0 bottom-0 z-20 pointer-events-none bg-primary w-[2px] transition-all duration-75"
              style={{
                left: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%',
              }}
            >
              <div className="absolute -top-[5px] -left-[4px] w-0 h-0 border-l-[5px] border-r-[5px] border-t-[8px] border-l-transparent border-r-transparent border-t-primary"></div>
            </div>

            {/* Tracks Container */}
            <div className="flex flex-col gap-[2px] py-[2px] relative z-0">
              {/* Grid Lines */}
              <div
                className="absolute inset-0 pointer-events-none opacity-10"
                style={{
                  backgroundImage: 'linear-gradient(90deg, #958ea0 1px, transparent 1px)',
                  backgroundSize: '50px 100%',
                }}
              ></div>

              {displaySpeakers.map((speaker, idx) => {
                const speakerSegs = displaySegments.filter((s) => s.speaker === speaker)
                const isSpeaker1 = idx % 2 === 0

                return (
                  <div key={speaker} className={`h-16 w-full relative ${!isSpeaker1 ? 'bg-primary/5' : ''}`}>
                    {speakerSegs.map((seg) => {
                      const isActiveClip = seg.segment_id === (currentActive?.segment_id ?? '')
                      return (
                        <div
                          key={seg.segment_id}
                          onClick={() => handleSelectSegment(seg)}
                          style={clipStyle(seg)}
                          className={`absolute h-12 top-2 rounded-sm overflow-hidden group cursor-pointer border ${
                            isActiveClip
                              ? 'bg-primary/20 border-primary ai-shimmer shadow-[0_0_10px_rgba(208,188,255,0.15)]'
                              : isSpeaker1
                              ? 'bg-secondary/10 border-secondary'
                              : 'bg-primary/10 border-primary/60'
                          }`}
                        >
                          <div className="absolute inset-0 flex items-center px-1">
                            <svg
                              className={`w-full h-8 ${isSpeaker1 ? 'text-secondary' : 'text-primary'}`}
                              fill="none"
                              preserveAspectRatio="none"
                              stroke="currentColor"
                              strokeLinecap="round"
                              strokeWidth="1.5"
                              viewBox="0 0 100 20"
                            >
                              <path d="M0,10 L5,2 L10,18 L15,5 L20,15 L25,8 L30,12 L35,4 L40,16 L45,10 L50,6 L55,14 L60,3 L65,17 L70,9 L75,11 L80,5 L85,15 L90,8 L95,12 L100,10"></path>
                            </svg>
                          </div>
                          <div className="absolute top-1 left-2 flex flex-col gap-0">
                            <span className="font-label-sm text-[8px] text-primary/80 truncate">
                              {seg.translation || seg.text}
                            </span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )
              })}

              {/* Track 3: Background Music */}
              <div className="h-16 w-full relative">
                <div className="absolute left-0 w-full h-12 top-2 bg-surface-container-highest/50 border border-outline-variant rounded-sm overflow-hidden opacity-50">
                  <div className="absolute inset-0 flex items-center px-1">
                    <svg className="w-full h-6 text-outline" fill="none" preserveAspectRatio="none" stroke="currentColor" strokeWidth="1" viewBox="0 0 200 20">
                      <path d="M0,10 Q10,5 20,10 T40,10 T60,10 T80,10 T100,10 T120,10 T140,10 T160,10 T180,10 T200,10"></path>
                    </svg>
                  </div>
                  <span className="absolute top-1 left-2 font-label-sm text-[8px] text-outline">Corporate_Bed_01.wav</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
