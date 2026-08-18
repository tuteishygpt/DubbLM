import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { ApiClient } from '../api/types'

// ── API types ─────────────────────────────────────────────────────────────
interface Job { id: string; status: string; project_name?: string | null }

interface VoiceProfile {
  tts_system: string
  model?: string
  voice_name?: string
  reference_mode?: string
}
interface ProfilesSnapshot { revision: string; profiles: Record<string, VoiceProfile> }

interface ProjectSummary {
  name: string
  display_name?: string
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

type NavigateView = 'HukFlow Studio' | 'Workflow' | 'Jobs' | 'Settings' | 'Voice Profiles' | 'Dubbing Texts'

export interface RunStepOption {
  id: string
  title: string
  badge: string
  description: string
  icon: string
}

export const RUN_STEPS: RunStepOption[] = [
  {
    id: 'tts_to_end',
    title: 'TTS to End (Re-dub & Assemble)',
    badge: 'Recommended',
    description: 'Re-generate speech for all segments and build final audio & video',
    icon: 'record_voice_over',
  },
  {
    id: 'combine_video',
    title: 'Combine Video Only (Mix & Subtitles)',
    badge: 'Fast',
    description: 'Re-mix existing audio with background and render video with subtitles',
    icon: 'movie_filter',
  },
  {
    id: 'translate_only',
    title: 'Re-translate (LLM Translation)',
    badge: 'LLM',
    description: 'Run context-aware LLM translation using current transcripts',
    icon: 'translate',
  },
  {
    id: 'transcribe_only',
    title: 'Transcribe Only (Diarization & STT)',
    badge: 'STT',
    description: 'Re-run speaker diarization and speech recognition only',
    icon: 'transcribe',
  },
  {
    id: 'from_scratch',
    title: 'From Scratch (Force Clear Cache)',
    badge: 'Full Run',
    description: 'Clear all caches and re-run entire dubbing pipeline from zero',
    icon: 'restart_alt',
  },
]

export interface StudioMediaInfo {
  jobId?: string | null
  projectName?: string | null
  videoSrc?: string | null
  audioSrc?: string | null
  files?: JobFile[]
}

export function HukFlowStudioView({
  client,
  onNavigate,
  pendingOpenJobId,
  onPendingOpenJobConsumed,
  onJobStarted,
  onActiveMediaChange,
  showSettingsPanel,
  onCloseSettingsPanel,
}: {
  client?: ApiClient
  onNavigate?: (view: NavigateView) => void
  pendingOpenJobId?: string | null
  onPendingOpenJobConsumed?: () => void
  onJobStarted?: (jobId: string, projectName?: string) => void
  onActiveMediaChange?: (info: StudioMediaInfo) => void
  showSettingsPanel?: boolean
  onCloseSettingsPanel?: () => void
}) {
  // ── voice assignment state ───────────────────────────────────────────────
  const [voiceProfiles, setVoiceProfiles] = useState<Record<string, VoiceProfile>>({})
  const [speakerMap, setSpeakerMap] = useState<Record<string, string>>({})
  const [speakerMapSaving, setSpeakerMapSaving] = useState(false)
  const [speakerMapStatus, setSpeakerMapStatus] = useState('')

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

  // ── Run Step action menu state ────────────────────────────────────────────
  const [showRunDropdown, setShowRunDropdown] = useState(false)
  const [isRunningStep, setIsRunningStep] = useState(false)
  const runDropdownRef = useRef<HTMLDivElement>(null)

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
  const [audioDurations, setAudioDurations] = useState<Record<string, number>>({})

  // ── studio tools state ───────────────────────────────────────────────────
  const [zoomLevel, setZoomLevel] = useState(50)
  const [mutedTracks, setMutedTracks] = useState<Record<string, boolean>>({ a1: true })
  const [searchQuery, setSearchQuery] = useState('')
  const [regeneratingId, setRegeneratingId] = useState<string | null>(null)
  const [playingAudioSegmentId, setPlayingAudioSegmentId] = useState<string | null>(null)
  const previewAudioRef = useRef<HTMLAudioElement | null>(null)
  const [statusMessage, setStatusMessage] = useState('')
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({})
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const timelineCanvasRef = useRef<HTMLDivElement>(null)

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

  // ── consume pendingOpenJobId from parent (Open Project sidebar action) ────
  useEffect(() => {
    if (!pendingOpenJobId) return
    setSelectedJobId(pendingOpenJobId)
    setSelectedProjectName('')
    onPendingOpenJobConsumed?.()
  }, [pendingOpenJobId]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── load jobs and ready projects ──────────────────────────────────────────
  useEffect(() => {
    if (!client) return
    // Load voice profiles for assignment dropdown
    client.get<ProfilesSnapshot>('/api/voice-profiles')
      .then((res) => setVoiceProfiles(res.profiles ?? {}))
      .catch(() => {})

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

    // 1b. Fetch speaker_map for the selected job
    client
      .get<{ speaker_map?: Record<string, string> }>(`/api/jobs/${encodeURIComponent(selectedJobId)}/speaker-map`)
      .then((res) => {
        if (res?.speaker_map && typeof res.speaker_map === 'object') {
          setSpeakerMap(res.speaker_map)
        }
      })
      .catch(() => {})

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

  // ── fetch actual duration of each segment's dubbed audio file ─────────────
  useEffect(() => {
    setAudioDurations({})
    if (!selectedJobId) return
    segments.forEach((seg) => {
      if (!seg.audio?.url) return
      const tempAudio = new Audio(seg.audio.url)
      const onLoaded = () => {
        if (Number.isFinite(tempAudio.duration) && tempAudio.duration > 0) {
          setAudioDurations((prev) => ({ ...prev, [seg.segment_id]: tempAudio.duration }))
        }
      }
      tempAudio.addEventListener('loadedmetadata', onLoaded)
    })
  }, [segments, selectedJobId])

  // ── inform parent about active media context for export ───────────────────
  useEffect(() => {
    onActiveMediaChange?.({
      jobId: selectedJobId || null,
      projectName: selectedProjectName || null,
      videoSrc,
      files: musicFiles,
    })
  }, [selectedJobId, selectedProjectName, videoSrc, musicFiles, onActiveMediaChange])

  // ── edit translation ───────────────────────────────────────────────────────
  const handleTranslationChange = (id: string, value: string) => {
    setSegments((prev) => prev.map((s) => s.segment_id === id ? { ...s, translation: value, synthesized_text: value } : s))
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
      // Load saved speaker_map from project metadata
      try {
        const detail = await client.get<{ saved_config?: { speaker_map?: Record<string, string> } }>(
          `/api/projects/${encodeURIComponent(projectName)}`,
        )
        const savedMap = detail?.saved_config?.speaker_map
        if (savedMap && typeof savedMap === 'object') {
          setSpeakerMap(savedMap)
        }
      } catch { /* ignore, speaker_map is optional */ }
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsLoading(false)
      setTimeout(() => setStatusMessage(''), 3000)
    }
  }

  // ── save voice assignment (speaker_map) ───────────────────────────────────
  const handleSaveSpeakerMap = async () => {
    if (!client) return
    const targetJobId = selectedJobId
    const projectName = selectedProjectName || projects.find((p) => p.job_id === selectedJobId)?.name || jobs.find((j) => j.id === selectedJobId)?.project_name
    if (!targetJobId && !projectName) {
      setSpeakerMapStatus('Open a project or job to save voice assignment.')
      setTimeout(() => setSpeakerMapStatus(''), 3000)
      return
    }
    setSpeakerMapSaving(true)
    setSpeakerMapStatus('Saving…')
    try {
      if (targetJobId) {
        await client.put(`/api/jobs/${encodeURIComponent(targetJobId)}/speaker-map`, { speaker_map: speakerMap })
      } else if (projectName) {
        await client.put(`/api/projects/${encodeURIComponent(projectName)}/speaker-map`, { speaker_map: speakerMap })
      }
      setSpeakerMapStatus('Voice assignment saved.')
    } catch (err) {
      setSpeakerMapStatus(err instanceof Error ? err.message : 'Failed to save.')
    } finally {
      setSpeakerMapSaving(false)
      setTimeout(() => setSpeakerMapStatus(''), 3000)
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
    setRegeneratingId(segmentId)
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
      setRegeneratingId(null)
      setTimeout(() => setStatusMessage(''), 3000)
    }
  }

  // ── audio preview for one segment ───────────────────────────────────────
  const handlePlaySegmentAudio = (segmentId: string, audioUrl: string) => {
    if (previewAudioRef.current) {
      previewAudioRef.current.pause()
      previewAudioRef.current = null
    }
    if (playingAudioSegmentId === segmentId) {
      setPlayingAudioSegmentId(null)
      return
    }
    try {
      const audio = new Audio(audioUrl)
      previewAudioRef.current = audio
      setPlayingAudioSegmentId(segmentId)
      audio.onended = () => {
        setPlayingAudioSegmentId(null)
        previewAudioRef.current = null
      }
      audio.onerror = () => {
        setPlayingAudioSegmentId(null)
        previewAudioRef.current = null
      }
      const playPromise = audio.play()
      if (playPromise !== undefined) {
        playPromise.catch(() => {
          setPlayingAudioSegmentId(null)
          previewAudioRef.current = null
        })
      }
    } catch {
      setPlayingAudioSegmentId(null)
      previewAudioRef.current = null
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

  // Zoom scale for voice and audio tracks (100% to 460% width)
  const trackWidthPercent = Math.max(100, Math.round(100 + (zoomLevel - 10) * 4))
  const rulerTicksCount = Math.max(5, Math.round(5 * (trackWidthPercent / 100)))

  // Auto-scroll timeline to follow playhead during playback when zoomed
  useEffect(() => {
    if (!timelineCanvasRef.current || duration <= 0) return
    const el = timelineCanvasRef.current
    if (el.scrollWidth > el.clientWidth && isPlaying) {
      const playheadX = (currentTime / duration) * el.scrollWidth
      const target = playheadX - el.clientWidth / 2
      el.scrollLeft = Math.max(0, target)
    }
  }, [currentTime, duration, isPlaying])

  const clipStyle = (s: Segment) => {
    const start = Math.min(Math.max(0, s.start), duration)
    // Use actual dubbed audio duration when available; fall back to original segment length
    const actualLen = audioDurations[s.segment_id]
    const clipLen = actualLen != null
      ? Math.min(actualLen, duration - start)
      : Math.min(Math.max(0, s.end - s.start), duration - start)
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

  // Close run dropdown when clicking outside
  useEffect(() => {
    if (!showRunDropdown) return
    const handle = (e: MouseEvent) => {
      if (runDropdownRef.current && !runDropdownRef.current.contains(e.target as Node)) {
        setShowRunDropdown(false)
      }
    }
    window.document.addEventListener('mousedown', handle)
    return () => window.document.removeEventListener('mousedown', handle)
  }, [showRunDropdown])

  const handleRunStep = async (stepId: string) => {
    if (!client) return
    setShowRunDropdown(false)
    setIsRunningStep(true)
    setStatusMessage(`Starting ${stepId}…`)

    try {
      // 1. If user has unsaved text changes, save them first!
      if (dirty && selectedJobId) {
        await handleSave()
      }

      let projectName = selectedProjectName
      if (!projectName && selectedJobId) {
        const foundProj = projects.find((p) => p.job_id === selectedJobId)
        if (foundProj) projectName = foundProj.name
      }

      if (projectName) {
        const res = await client.post<{ project_name: string; job: Job }>(
          `/api/projects/${encodeURIComponent(projectName)}/run`,
          { run_step: stepId, overrides: {} },
        )
        if (res?.job?.id) {
          setStatusMessage(`Started ${stepId} (Job: ${res.job.id})`)
          onJobStarted?.(res.job.id, projectName)
        }
      } else {
        setLoadError('Please select or open a project first to run pipeline steps.')
      }
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Failed to start pipeline step.')
    } finally {
      setIsRunningStep(false)
    }
  }

  const [portalTarget, setPortalTarget] = useState<HTMLElement | null>(null)

  useEffect(() => {
    const el = window.document.getElementById('studio-header-run-slot')
    if (el) setPortalTarget(el)
  }, [])

  const runMenuNode = (selectedProjectName || selectedJobId) ? (
    <div className="studio-run-menu-wrap" ref={runDropdownRef}>
      <button
        type="button"
        className="studio-run-btn"
        onClick={() => setShowRunDropdown((v) => !v)}
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

      {/* Render Run Step menu next to Export button via portal if header slot exists */}
      {portalTarget && runMenuNode ? createPortal(runMenuNode, portalTarget) : null}

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

              {/* Fallback inline render if portal target is not present */}
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
                const speakerIdx = displaySpeakers.indexOf(seg.speaker)
                const speakerSlot = SPEAKER_COLOURS[speakerIdx % 4]
                const isSourceExpanded = Boolean(expandedSources[seg.segment_id])
                // Calculate dynamic row count based on text length and newlines
                const explicitLines = seg.translation.split('\n').length
                const estimatedLines = Math.ceil((seg.translation.length || 1) / 75)
                const dynamicRows = Math.max(1, Math.min(8, Math.max(explicitLines, estimatedLines)))

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
                        <span className={`speaker-dot speaker-dot-${speakerIdx % 4}`}></span>
                        <span className="speaker-name">{seg.speaker}</span>
                        <span className="time-badge">
                          {formatTime(seg.start)} – {formatTime(seg.end)}
                        </span>
                        {visibleSegments.length > 1 && (
                          <span className="segment-stepper-label">
                            ({idx + 1}/{visibleSegments.length})
                          </span>
                        )}
                        <span className="dubbed-tag">DUBBED (BE)</span>
                        <span className="text-count-badge">{seg.translation.length} chars</span>
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
                          <button
                            type="button"
                            className={`audio-play-btn ${playingAudioSegmentId === seg.segment_id ? 'playing' : ''}`}
                            title={playingAudioSegmentId === seg.segment_id ? `Stop ${seg.audio.name}` : `Play ${seg.audio.name}`}
                            data-testid={`play-btn-${seg.segment_id}`}
                            aria-label={playingAudioSegmentId === seg.segment_id ? `Stop audio for ${seg.speaker}` : `Play audio for ${seg.speaker}`}
                            onClick={(e) => {
                              e.stopPropagation()
                              handlePlaySegmentAudio(seg.segment_id, seg.audio!.url)
                            }}
                          >
                            <span className="material-symbols-outlined">
                              {playingAudioSegmentId === seg.segment_id ? 'pause_circle' : 'play_circle'}
                            </span>
                          </button>
                        )}

                        <button
                          type="button"
                          className={`regenerate-btn ${regeneratingId === seg.segment_id ? 'spinning' : ''}`}
                          onClick={(e) => {
                            e.stopPropagation()
                            void handleRegenerateAudio(seg.segment_id)
                          }}
                          title={`Regenerate Audio for ${seg.speaker}`}
                          data-testid={`regenerate-btn-${seg.segment_id}`}
                          aria-label={`Regenerate Audio for ${seg.speaker}`}
                          disabled={Boolean(regeneratingId)}
                        >
                          <span className="material-symbols-outlined">autorenew</span>
                        </button>
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
                    <div className="editable-dubbed-wrapper">
                      <textarea
                        className="dubbed-textarea"
                        value={seg.synthesized_text || seg.translation}
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
                    </div>
                  </div>
                )
              })}
          </div>
        </div>

        {/* RIGHT PANE: Video Player + Voice Assignment ────────────────────── */}
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
        {/* Floating semi-transparent zoom controls */}
        <div className="timeline-floating-zoom" onClick={(e) => e.stopPropagation()}>
          <button
            type="button"
            className="zoom-btn"
            title="Zoom Out"
            onClick={() => setZoomLevel((z) => Math.max(10, z - 10))}
            aria-label="Zoom Out"
          >
            <span className="material-symbols-outlined">zoom_out</span>
          </button>
          <input
            type="range"
            min="10"
            max="100"
            value={zoomLevel}
            onChange={(e) => setZoomLevel(Number(e.target.value))}
            className="zoom-slider"
            aria-label="Zoom timeline"
          />
          <button
            type="button"
            className="zoom-btn"
            title="Zoom In"
            onClick={() => setZoomLevel((z) => Math.min(100, z + 10))}
            aria-label="Zoom In"
          >
            <span className="material-symbols-outlined">zoom_in</span>
          </button>
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

          {/* Timeline canvas with horizontal scroll & zoom */}
          <div
            ref={timelineCanvasRef}
            className="timeline-tracks-canvas"
          >
            <div
              className="timeline-tracks-inner"
              style={{
                width: `${trackWidthPercent}%`,
                minWidth: `${trackWidthPercent}%`,
              }}
              onClick={(e) => {
                if ((e.target as HTMLElement).closest('.clip-block')) return
                const rect = e.currentTarget.getBoundingClientRect()
                const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
                seekToTime(pct * duration)
              }}
            >
              {/* Time ruler */}
              <div className="time-ruler">
                {Array.from({ length: rulerTicksCount }, (_, i) => (
                  <span
                    key={i}
                    style={{
                      position: 'absolute',
                      left: `${(i / (rulerTicksCount - 1)) * 100}%`,
                      transform: i === 0 ? 'none' : i === rulerTicksCount - 1 ? 'translateX(-100%)' : 'translateX(-50%)',
                    }}
                  >
                    {formatTime((duration / (rulerTicksCount - 1)) * i)}
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
      </div>

      {/* ── Settings Modal (Voice Assignment) ─────────────────────────────── */}
      {showSettingsPanel && (
        <>
          {/* Overlay */}
          <div
            className="settings-modal-overlay"
            onClick={onCloseSettingsPanel}
            aria-hidden="true"
          />
          {/* Drawer panel */}
          <div className="settings-modal-panel" role="dialog" aria-label="Project Settings" aria-modal="true">
            <div className="settings-modal-header">
              <div className="settings-modal-title">
                <span className="material-symbols-outlined">settings</span>
                Project Settings
              </div>
              <button
                type="button"
                className="settings-modal-close"
                onClick={onCloseSettingsPanel}
                aria-label="Close settings"
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>

            <div className="settings-modal-body">
              {/* ── Voice Assignment ── */}
              <div className="settings-section">
                <div className="settings-section-title">
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>manage_accounts</span>
                  Voice Assignment
                </div>
                <p className="settings-section-desc">
                  Map each detected speaker to a voice profile from the global library.
                  Changes apply to the next pipeline run.
                </p>

                {displaySpeakers.length === 0 ? (
                  <div className="settings-empty">
                    <span className="material-symbols-outlined" style={{ fontSize: '28px', color: 'var(--text-muted)' }}>person_off</span>
                    <span>No speakers detected. Open a project first.</span>
                  </div>
                ) : (
                  <div className="voice-assignment-rows">
                    {displaySpeakers.map((speaker, idx) => (
                      <div key={speaker} className="voice-assignment-row">
                        <span className={`speaker-dot speaker-dot-${idx % 4}`} style={{ flexShrink: 0 }} />
                        <span className="voice-assignment-speaker">{speaker}</span>
                        <select
                          className="voice-assignment-select"
                          value={speakerMap[speaker] ?? ''}
                          onChange={(e) => setSpeakerMap((prev) => ({ ...prev, [speaker]: e.target.value }))}
                          aria-label={`Voice profile for ${speaker}`}
                        >
                          <option value="">— no override —</option>
                          {Object.keys(voiceProfiles).map((name) => (
                            <option key={name} value={name}>{name}</option>
                          ))}
                        </select>
                      </div>
                    ))}
                  </div>
                )}

                {speakerMapStatus && (
                  <div className={`settings-status-msg${speakerMapStatus.includes('fail') || speakerMapStatus.includes('Open') ? ' settings-status-msg--err' : ''}`}>
                    {speakerMapStatus}
                  </div>
                )}

                <div className="settings-modal-actions">
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={handleSaveSpeakerMap}
                    disabled={speakerMapSaving || displaySpeakers.length === 0}
                  >
                    {speakerMapSaving
                      ? <><span className="material-symbols-outlined spin-anim" style={{ fontSize: '15px' }}>progress_activity</span> Saving…</>
                      : <><span className="material-symbols-outlined" style={{ fontSize: '15px' }}>save</span> Save assignment</>}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  )
}
