import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { ApiClient, ConfigResponse, OptionsResponse, JsonValue, SelectOption, SchemaFieldDefinition } from '../api/types'
import { SchemaField } from '../components/SchemaField'

// ── API types ─────────────────────────────────────────────────────────────
import type { Job, VoiceProfile, ProfilesSnapshot, ProjectSummary, JobFile, SegmentAudio, Segment, TextDocument } from '../types/models'

const SPEAKER_COLOURS = ['speaker-0', 'speaker-1', 'speaker-2', 'speaker-3'] as const



import { SegmentWaveform } from '../components/studio/SegmentWaveform'

import { formatTime, formatTimecode, parseTimeInput } from '../utils/time'

function normalizeOptions(value: unknown): SelectOption[] {
  return Array.isArray(value) ? value.map((item) => typeof item === 'string' ? { value: item, label: item } : item as SelectOption) : []
}

function getFieldSection(fieldName: string): string {
  if (fieldName.includes('llm') || fieldName.includes('translator') || fieldName.includes('refinement') || fieldName === 'translation_prompt_prefix' || fieldName === 'glossary') return 'LLM & Translation';
  if (fieldName.includes('transcription')) return 'Transcription & Diarization';
  if (fieldName.includes('tts') || fieldName.includes('voice') || fieldName === 'segment_reference_min_duration') return 'Voice Generation (TTS)';
  if (fieldName.includes('emotion')) return 'Emotions & Context';
  if (fieldName.includes('timing') || fieldName.includes('volume') || fieldName.includes('pause') || fieldName.includes('semantic_split') || fieldName === 'keyframe_buffer' || fieldName === 'use_two_pass_encoding' || fieldName === 'keep_original_audio_ranges') return 'Timing & Mixing';
  if (fieldName.includes('debug') || fieldName.includes('watermark')) return 'Debug & Watermarks';
  return 'General Execution';
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

  // ── global config state ──────────────────────────────────────────────────
  const [globalConfig, setGlobalConfig] = useState<ConfigResponse | null>(null)
  const [globalOptions, setGlobalOptions] = useState<OptionsResponse>({})
  const [globalSettingsValues, setGlobalSettingsValues] = useState<Record<string, JsonValue>>({})
  const [globalSettingsSaving, setGlobalSettingsSaving] = useState(false)
  const [globalSettingsStatus, setGlobalSettingsStatus] = useState('')
  const [activeSettingsTab, setActiveSettingsTab] = useState<string>('all')

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

    client.get<ConfigResponse>('/api/config')
      .then(res => {
        setGlobalConfig(res)
        setGlobalSettingsValues(res.values)
      })
      .catch(() => {})

    client.get<OptionsResponse>('/api/options')
      .then(setGlobalOptions)
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
    const AudioContextClass =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    
    segments.forEach((seg) => {
      if (!seg.audio?.url) return
      
      if (AudioContextClass) {
        const ctx = new AudioContextClass()
        fetch(seg.audio.url)
          .then((r) => r.arrayBuffer())
          .then((buf) => ctx.decodeAudioData(buf))
          .then((decoded) => {
            if (decoded.duration > 0) {
              setAudioDurations((prev) => ({ ...prev, [seg.segment_id]: decoded.duration }))
            }
          })
          .catch((err) => console.warn('Failed to decode audio duration:', err))
      } else {
        // Fallback for environments without AudioContext
        const tempAudio = new Audio(seg.audio.url)
        const onLoaded = () => {
          if (Number.isFinite(tempAudio.duration) && tempAudio.duration > 0) {
            setAudioDurations((prev) => ({ ...prev, [seg.segment_id]: tempAudio.duration }))
          }
        }
        if (tempAudio.readyState >= 1) {
          onLoaded()
        } else {
          tempAudio.addEventListener('loadedmetadata', onLoaded)
        }
      }
    })
  }, [segments, selectedJobId])

  // ── inform parent about active media context for export ───────────────────
  useEffect(() => {
    const resolvedProjectName =
      selectedProjectName ||
      projects.find((p) => p.job_id === selectedJobId)?.name ||
      jobs.find((j) => j.id === selectedJobId)?.project_name ||
      null
    onActiveMediaChange?.({
      jobId: selectedJobId || null,
      projectName: resolvedProjectName,
      videoSrc,
      files: musicFiles,
    })
  }, [selectedJobId, selectedProjectName, projects, jobs, videoSrc, musicFiles, onActiveMediaChange])

  // ── edit translation ───────────────────────────────────────────────────────
  const handleTranslationChange = (id: string, value: string) => {
    setSegments((prev) => prev.map((s) => s.segment_id === id ? { ...s, translation: value, synthesized_text: value } : s))
    setDirty(true)
  }

  // ── edit speaker ──────────────────────────────────────────────────────────
  const handleSpeakerChange = (id: string, value: string) => {
    setSegments((prev) => prev.map((s) => s.segment_id === id ? { ...s, speaker: value } : s))
    setDirty(true)
  }

  // ── edit timestamps ───────────────────────────────────────────────────────
  const handleTimestampChange = (id: string, field: 'start' | 'end', value: number) => {
    setSegments((prev) => prev.map((s) => s.segment_id === id ? { ...s, [field]: value } : s))
    setDirty(true)
  }

  // ── edit original (source) text ───────────────────────────────────────────
  const handleSourceTextChange = (id: string, value: string) => {
    setSegments((prev) => prev.map((s) => s.segment_id === id ? { ...s, text: value } : s))
    setDirty(true)
  }

  // ── delete a single segment ───────────────────────────────────────────────
  const handleDeleteSegment = (id: string) => {
    const seg = segments.find((s) => s.segment_id === id)
    if (!seg) return
    setSegments((prev) => prev.filter((s) => s.segment_id !== id))
    if (activeSegmentId === id) setActiveSegmentId('')
    setDirty(true)
    setStatusMessage('Segment deleted. Save to apply.')
    setTimeout(() => setStatusMessage(''), 4000)
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
  useEffect(() => {
    return () => {
      if (previewAudioRef.current) {
        previewAudioRef.current.pause()
        previewAudioRef.current = null
      }
    }
  }, [])

  const handlePlaySegmentAudio = (segmentId: string, audioUrl: string) => {
    if (previewAudioRef.current) {
      previewAudioRef.current.pause()
      previewAudioRef.current = null
    }
    if (playingAudioSegmentId === segmentId) {
      setPlayingAudioSegmentId(null)
      return
    }
    // Pause main video if it's playing so audios don't overlap
    if (isPlaying) {
      setIsPlaying(false)
    }
    if (videoRef.current && !videoRef.current.paused) {
      videoRef.current.pause()
    }
    try {
      // Append a cache-buster so that if the audio was regenerated, we don't play the cached version
      const urlWithBuster = audioUrl.includes('?') ? `${audioUrl}&t=${Date.now()}` : `${audioUrl}?t=${Date.now()}`
      const audio = new Audio(urlWithBuster)
      previewAudioRef.current = audio
      setPlayingAudioSegmentId(segmentId)
      audio.onended = () => {
        if (previewAudioRef.current === audio) {
          setPlayingAudioSegmentId(null)
          previewAudioRef.current = null
        }
      }
      audio.onerror = () => {
        if (previewAudioRef.current === audio) {
          setPlayingAudioSegmentId(null)
          previewAudioRef.current = null
        }
      }
      const playPromise = audio.play()
      if (playPromise !== undefined) {
        playPromise.catch(() => {
          if (previewAudioRef.current === audio) {
            setPlayingAudioSegmentId(null)
            previewAudioRef.current = null
          }
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
      // Pause segment audio preview if it's playing
      if (previewAudioRef.current) {
        previewAudioRef.current.pause()
        previewAudioRef.current = null
        setPlayingAudioSegmentId(null)
      }
      
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

      const projectName =
        selectedProjectName ||
        projects.find((p) => p.job_id === selectedJobId)?.name ||
        jobs.find((j) => j.id === selectedJobId)?.project_name ||
        ''

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
                        <select
                          className="speaker-name-select"
                          value={seg.speaker}
                          onChange={(e) => { e.stopPropagation(); handleSpeakerChange(seg.segment_id, e.target.value) }}
                          onClick={(e) => e.stopPropagation()}
                          title="Change speaker"
                        >
                          {displaySpeakers.map((sp) => (
                            <option key={sp} value={sp}>{sp}</option>
                          ))}
                        </select>
                        <span className="time-badge editable-time-badge" onClick={(e) => e.stopPropagation()}>
                          <input
                            className="time-input"
                            defaultValue={formatTime(seg.start)}
                            key={`start-${seg.segment_id}-${seg.start}`}
                            onBlur={(e) => {
                              const v = parseTimeInput(e.target.value)
                              if (v !== null) handleTimestampChange(seg.segment_id, 'start', v)
                              else e.target.value = formatTime(seg.start)
                            }}
                            onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
                            title="Start time (MM:SS)"
                          />
                          <span className="time-separator"> – </span>
                          <input
                            className="time-input"
                            defaultValue={formatTime(seg.end)}
                            key={`end-${seg.segment_id}-${seg.end}`}
                            onBlur={(e) => {
                              const v = parseTimeInput(e.target.value)
                              if (v !== null) handleTimestampChange(seg.segment_id, 'end', v)
                              else e.target.value = formatTime(seg.end)
                            }}
                            onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
                            title="End time (MM:SS)"
                          />
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

                        <button
                          type="button"
                          className="delete-segment-btn"
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteSegment(seg.segment_id)
                          }}
                          title={`Delete segment ${seg.speaker}`}
                          data-testid={`delete-btn-${seg.segment_id}`}
                          aria-label={`Delete segment ${seg.speaker}`}
                        >
                          <span className="material-symbols-outlined">delete</span>
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
                        <textarea
                          className={`source-text-textarea source-text-${speakerSlot.replace('speaker-', '')}`}
                          value={seg.text}
                          onChange={(e) => handleSourceTextChange(seg.segment_id, e.target.value)}
                          rows={2}
                          aria-label={`Source text for ${seg.speaker}`}
                        />
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
                          <SegmentWaveform url={seg.audio?.url ?? null} />
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
            {/* Header */}
            <div className="settings-modal-header">
              <div className="settings-modal-title">
                <span className="material-symbols-outlined" style={{ color: 'var(--primary)' }}>tune</span>
                <span>Project & Pipeline Settings</span>
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

            {/* Quick Filter Tabs */}
            <div className="settings-modal-tabs" role="tablist">
              {[
                { id: 'all', label: 'All Stages', icon: 'apps' },
                { id: 'voices', label: 'Voices', icon: 'manage_accounts' },
                { id: 'LLM & Translation', label: 'LLM & Trans', icon: 'psychology' },
                { id: 'Transcription & Diarization', label: 'Transcription', icon: 'subtitles' },
                { id: 'Voice Generation (TTS)', label: 'TTS Engine', icon: 'record_voice_over' },
                { id: 'Timing & Mixing', label: 'Timing & Mix', icon: 'tune' },
                { id: 'Emotions & Context', label: 'Emotions', icon: 'mood' },
                { id: 'Debug & Watermarks', label: 'Debug', icon: 'bug_report' },
              ].map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  aria-selected={activeSettingsTab === tab.id}
                  className={`settings-tab-btn${activeSettingsTab === tab.id ? ' active' : ''}`}
                  onClick={() => setActiveSettingsTab(tab.id)}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>{tab.icon}</span>
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Modal Body */}
            <div className="settings-modal-body">
              {/* ── Voice Assignment Section ── */}
              {(activeSettingsTab === 'all' || activeSettingsTab === 'voices') && (
                <div className="settings-group-card settings-group-card--voice">
                  <div className="settings-group-title">
                    <span className="material-symbols-outlined">manage_accounts</span>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flex: 1 }}>
                      <span>Voice Assignment</span>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'none', fontWeight: 500 }}>
                        {displaySpeakers.length} Detected Speakers
                      </span>
                    </div>
                  </div>
                  <p className="settings-section-desc" style={{ marginBottom: '1rem' }}>
                    Map each detected speaker to a voice profile from the global library.
                  </p>

                  {displaySpeakers.length === 0 ? (
                    <div className="settings-empty">
                      <span className="material-symbols-outlined" style={{ fontSize: '28px', color: 'var(--text-muted)' }}>person_off</span>
                      <span>No speakers detected. Open a project first.</span>
                    </div>
                  ) : (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.75rem' }}>
                      {displaySpeakers.map((speaker, idx) => (
                        <div
                          key={speaker}
                          style={{
                            background: 'var(--bg-subtle, #1c1b1b)',
                            border: '1px solid rgba(255,255,255,0.06)',
                            borderRadius: '8px',
                            padding: '0.625rem 0.875rem',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.75rem',
                          }}
                        >
                          <span className={`speaker-dot speaker-dot-${idx % 4}`} style={{ width: '8px', height: '8px', borderRadius: '50%', flexShrink: 0 }} />
                          <span className="voice-assignment-speaker" style={{ fontWeight: 600, minWidth: '85px' }}>{speaker}</span>
                          <select
                            className="voice-assignment-select"
                            style={{ flex: 1, padding: '0.35rem 0.5rem', fontSize: '0.8125rem' }}
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
                    <div className={`settings-status-msg${speakerMapStatus.includes('fail') || speakerMapStatus.includes('Open') ? ' settings-status-msg--err' : ''}`} style={{ marginTop: '0.75rem' }}>
                      {speakerMapStatus}
                    </div>
                  )}
                </div>
              )}
              
              {/* ── Pipeline Parameters ── */}
              {globalConfig && (() => {
                const grouped = globalConfig.schema.fields.reduce((acc, field) => {
                  if (field.type === 'list' || field.type === 'object' || field.type === 'structured') return acc;
                  const sec = field.section || getFieldSection(field.name);
                  if (!acc[sec]) acc[sec] = [];
                  acc[sec].push(field);
                  return acc;
                }, {} as Record<string, SchemaFieldDefinition[]>);

                const filteredGroups = Object.entries(grouped).filter(([secName]) => {
                  if (activeSettingsTab === 'all') return true;
                  return activeSettingsTab === secName;
                });

                return (
                  <div style={{ display: 'grid', gridTemplateColumns: filteredGroups.length === 1 ? '1fr' : 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1rem' }}>
                    {filteredGroups.map(([secName, fields]) => {
                      let icon = 'settings_input_component';
                      let cardClass = 'settings-group-card--general';
                      if (secName.includes('LLM') || secName.includes('Translation')) { icon = 'psychology'; cardClass = 'settings-group-card--llm'; }
                      else if (secName.includes('Transcription')) { icon = 'subtitles'; cardClass = 'settings-group-card--transcription'; }
                      else if (secName.includes('Voice')) { icon = 'record_voice_over'; cardClass = 'settings-group-card--voice'; }
                      else if (secName.includes('Emotion')) { icon = 'mood'; cardClass = 'settings-group-card--emotion'; }
                      else if (secName.includes('Timing') || secName.includes('Mixing')) { icon = 'tune'; cardClass = 'settings-group-card--timing'; }
                      else if (secName.includes('Debug')) { icon = 'bug_report'; cardClass = 'settings-group-card--debug'; }
                      else if (secName.includes('Execution')) { icon = 'power_settings_new'; cardClass = 'settings-group-card--general'; }

                      return (
                        <div key={secName} className={`settings-group-card ${cardClass}`}>
                          <div className="settings-group-title">
                            <span className="material-symbols-outlined">{icon}</span>
                            <span>{secName}</span>
                          </div>
                          <div className="settings-field-flow">
                            {fields.map(field => (
                              <SchemaField
                                key={field.name}
                                field={field}
                                value={globalSettingsValues[field.name]}
                                options={normalizeOptions(field.options ?? globalOptions[field.options_key ?? field.name])}
                                onChange={(value) => setGlobalSettingsValues((prev) => ({ ...prev, [field.name]: value }))}
                              />
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                );
              })()}
            </div>

            {/* Sticky Action Footer */}
            <div className="settings-modal-footer">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {globalSettingsStatus && (
                  <span className={`settings-status-msg${globalSettingsStatus.includes('fail') ? ' settings-status-msg--err' : ''}`} style={{ margin: 0 }}>
                    {globalSettingsStatus}
                  </span>
                )}
              </div>
              <div className="settings-modal-footer-actions">
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={onCloseSettingsPanel}
                  style={{ padding: '0.45rem 1rem' }}
                >
                  Close
                </button>
                <button
                  type="button"
                  className="btn-primary"
                  onClick={async () => {
                    if (displaySpeakers.length > 0) {
                      handleSaveSpeakerMap();
                    }
                    if (client && globalConfig) {
                      setGlobalSettingsSaving(true);
                      try {
                        const normalized = { ...globalSettingsValues };
                        for (const field of globalConfig.schema.fields) {
                          if (['list', 'object', 'structured'].includes(field.type) && typeof normalized[field.name] === 'string') {
                            try { normalized[field.name] = JSON.parse(normalized[field.name] as string) as JsonValue; } catch (e) {}
                          }
                        }
                        const updated = await client.put<ConfigResponse>('/api/config', { revision: globalConfig.revision, values: normalized });
                        
                        const projectName = selectedProjectName || projects.find((p) => p.job_id === selectedJobId)?.name || jobs.find((j) => j.id === selectedJobId)?.project_name;
                        if (projectName) {
                          try {
                            await client.put(`/api/projects/${encodeURIComponent(projectName)}/config`, { values: normalized });
                          } catch (err) {
                            console.warn('Failed to sync settings to project metadata:', err);
                          }
                        }

                        setGlobalConfig(updated);
                        setGlobalSettingsValues(updated.values);
                        setGlobalSettingsStatus(projectName ? 'Settings saved to global and active project.' : 'Settings applied & saved.');
                      } catch (e) {
                        setGlobalSettingsStatus(e instanceof Error ? e.message : 'Failed to save.');
                      } finally {
                        setGlobalSettingsSaving(false);
                        setTimeout(() => setGlobalSettingsStatus(''), 3500);
                      }
                    }
                  }}
                  disabled={globalSettingsSaving || speakerMapSaving}
                  style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', padding: '0.45rem 1.25rem' }}
                >
                  {globalSettingsSaving || speakerMapSaving
                    ? <><span className="material-symbols-outlined spin-anim" style={{ fontSize: '16px' }}>progress_activity</span> Applying…</>
                    : <><span className="material-symbols-outlined" style={{ fontSize: '16px' }}>bolt</span> Save & Apply</>}
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  )
}
