import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { ApiClient, ConfigResponse, OptionsResponse, JsonValue, SelectOption, SchemaFieldDefinition } from '../api/types'
import { SchemaField } from '../components/SchemaField'

// ── API types ─────────────────────────────────────────────────────────────
import type { Job, VoiceProfile, ProfilesSnapshot, ProjectSummary, JobFile, SegmentAudio, Segment, TextDocument } from '../types/models'



import { parseTimeInput } from '../utils/time'
export type NavigateView = 'HukFlow Studio' | 'Workflow' | 'Jobs' | 'Settings' | 'Voice Profiles' | 'Dubbing Texts'

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

import { createContext, useContext, ReactNode } from 'react'

export interface StudioState {
  client?: ApiClient
  onNavigate?: (view: NavigateView) => void
  pendingOpenJobId?: string | null
  onPendingOpenJobConsumed?: () => void
  onJobStarted?: (jobId: string, projectName?: string) => void
  onActiveMediaChange?: (info: StudioMediaInfo) => void
  showSettingsPanel?: boolean
  onCloseSettingsPanel?: () => void
  [key: string]: any // Temporarily allow any for remaining context values
  displaySpeakers: string[]
  displaySegments: Segment[]
  visibleSegments: Segment[]
  projects: ProjectSummary[]
  jobs: Job[]
  selectedJobId: string
  selectedProjectName: string
  showRunDropdown: boolean
  setShowRunDropdown: React.Dispatch<React.SetStateAction<boolean>>
  isRunningStep: boolean
  handleRunStep: (stepId: string) => Promise<void>
  handleOpenProject: (projName: string) => Promise<void>
  setSelectedJobId: React.Dispatch<React.SetStateAction<string>>
  setSelectedProjectName: React.Dispatch<React.SetStateAction<string>>
  dirty: boolean
  handleSave: () => Promise<void>
  searchQuery: string
  setSearchQuery: React.Dispatch<React.SetStateAction<string>>
  statusMessage: string
  loadError: string
  scrollAreaRef: React.RefObject<HTMLDivElement>
  isLoading: boolean
  currentActive: Segment | null
  expandedSources: Record<string, boolean>
  portalTarget: HTMLElement | null
  runDropdownRef: React.RefObject<HTMLDivElement>
  playingAudioSegmentId: string | null
  regeneratingId: string | null
  handleSelectSegment: (seg: Segment) => void
  handleSpeakerChange: (id: string, val: string) => void
  handleTimestampChange: (id: string, field: 'start' | 'end', val: number) => void
  handleSourceTextChange: (id: string, val: string) => void
  handleTranslationChange: (id: string, val: string) => void
  toggleSource: (id: string) => void
  handlePlaySegmentAudio: (id: string, url: string) => void
  handleRegenerateAudio: (id: string) => void
  handleDeleteSegment: (id: string) => void
  videoRef: React.RefObject<HTMLVideoElement>
  videoSrc: string
  currentTime: number
  setCurrentTime: React.Dispatch<React.SetStateAction<number>>
  duration: number
  videoDuration: number
  setVideoDuration: React.Dispatch<React.SetStateAction<number>>
  isPlaying: boolean
  setIsPlaying: React.Dispatch<React.SetStateAction<boolean>>
  togglePlayPause: () => void
  liveSubSegment: Segment | undefined
  handleScrubberClick: (e: React.MouseEvent<HTMLDivElement>) => void
  musicFiles: JobFile[]
  backgroundDurations: Record<string, number>
  audioDurations: Record<string, number>
  mutedTracks: Record<string, boolean>
  toggleTrackMute: (track: string) => void
  zoomLevel: number
  setZoomLevel: React.Dispatch<React.SetStateAction<number>>
  seekToTime: (time: number) => void
  activeSettingsTab: string
  setActiveSettingsTab: React.Dispatch<React.SetStateAction<string>>
  voiceProfiles: Record<string, VoiceProfile>
  speakerMap: Record<string, string>
  setSpeakerMap: React.Dispatch<React.SetStateAction<Record<string, string>>>
  speakerMapStatus: string
  globalConfig: ConfigResponse | null
  globalOptions: OptionsResponse
  globalSettingsValues: Record<string, JsonValue>
  setGlobalSettingsValues: React.Dispatch<React.SetStateAction<Record<string, JsonValue>>>
  globalSettingsSaving: boolean
  globalSettingsStatus: string
  handleSaveSpeakerMap: () => Promise<void>
  setGlobalSettingsStatus: React.Dispatch<React.SetStateAction<string>>
  setGlobalConfig: React.Dispatch<React.SetStateAction<ConfigResponse | null>>
  setGlobalSettingsSaving: React.Dispatch<React.SetStateAction<boolean>>
  speakerMapSaving: boolean
}

const StudioContext = createContext<StudioState | null>(null)

export function useStudio() {
  const ctx = useContext(StudioContext)
  if (!ctx) throw new Error('useStudio must be used within a StudioProvider')
  return ctx
}

export function StudioProvider({
  client,
  onNavigate,
  pendingOpenJobId,
  onPendingOpenJobConsumed,
  onJobStarted,
  onActiveMediaChange,
  showSettingsPanel,
  onCloseSettingsPanel,
  children
}: {
  client?: ApiClient
  onNavigate?: (view: NavigateView) => void
  pendingOpenJobId?: string | null
  onPendingOpenJobConsumed?: () => void
  onJobStarted?: (jobId: string, projectName?: string) => void
  onActiveMediaChange?: (info: StudioMediaInfo) => void
  showSettingsPanel?: boolean
  onCloseSettingsPanel?: () => void
  children: ReactNode
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

  // ── context value ────────────────────────────────────────────────────────
  const value = {
    client,
    onNavigate,
    pendingOpenJobId,
    onPendingOpenJobConsumed,
    onJobStarted,
    onActiveMediaChange,
    showSettingsPanel,
    onCloseSettingsPanel,
    voiceProfiles, setVoiceProfiles,
    speakerMap, setSpeakerMap,
    speakerMapSaving, setSpeakerMapSaving,
    speakerMapStatus, setSpeakerMapStatus,
    globalConfig, setGlobalConfig,
    globalOptions, setGlobalOptions,
    globalSettingsValues, setGlobalSettingsValues,
    globalSettingsSaving, setGlobalSettingsSaving,
    globalSettingsStatus, setGlobalSettingsStatus,
    activeSettingsTab, setActiveSettingsTab,
    jobs, setJobs,
    projects, setProjects,
    selectedProjectName, setSelectedProjectName,
    selectedJobId, setSelectedJobId,
    document, setDocument,
    segments, setSegments,
    revision, setRevision,
    dirty, setDirty,
    loadError, setLoadError,
    isLoading, setIsLoading,
    showRunDropdown, setShowRunDropdown,
    isRunningStep, setIsRunningStep,
    runDropdownRef,
    activeSegmentId, setActiveSegmentId,
    videoRef,
    videoSrc, setVideoSrc,
    isPlaying, setIsPlaying,
    currentTime, setCurrentTime,
    videoDuration, setVideoDuration,
    musicFiles, setMusicFiles,
    backgroundDurations, setBackgroundDurations,
    audioDurations, setAudioDurations,
    zoomLevel, setZoomLevel,
    mutedTracks, setMutedTracks,
    searchQuery, setSearchQuery,
    regeneratingId, setRegeneratingId,
    playingAudioSegmentId, setPlayingAudioSegmentId,
    previewAudioRef,
    statusMessage, setStatusMessage,
    expandedSources, setExpandedSources,
    scrollAreaRef,
    timelineCanvasRef,
    toggleSource,
    isStandaloneDemo,
    displaySegments,
    displaySpeakers,
    visibleSegments,
    currentActive,
    liveSubSegment,
    duration,
    handleSave,
    handleOpenProject,
    handleRunStep,
    handleSelectSegment,
    togglePlayPause,
    seekToTime,
    handleScrubberClick,
    handleTimestampChange,
    handleSourceTextChange,
    handleTranslationChange,
    handleSpeakerChange,
    handleDeleteSegment,
    handleSaveSpeakerMap,
    handlePlaySegmentAudio,
    handleRegenerateAudio,
    toggleTrackMute,
    portalTarget, setPortalTarget
  }

  return <StudioContext.Provider value={value}>{children}</StudioContext.Provider>
}
