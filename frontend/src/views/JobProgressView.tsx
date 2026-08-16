import { useEffect, useMemo, useRef, useState } from 'react'
import type { ApiClient, ConnectionState, SseEvent } from '../api/types'

export interface PipelineStage {
  id: string
  number: number
  title: string
  subtitle: string
  description: string
  icon: string
  patterns: RegExp[]
  baseProgress: number
  maxProgress: number
}

export const PIPELINE_STAGES: PipelineStage[] = [
  {
    id: 'audio_prep',
    number: 1,
    title: 'Audio Preparation & Extraction',
    subtitle: 'Extract & Separate',
    description: 'Extracting source audio track from video, separating background music and noise from vocal tracks.',
    icon: 'audio_file',
    patterns: [
      /starting dubbing process/i,
      /separating audio/i,
      /extracting audio/i,
      /demucs/i,
      /prepare_audio/i,
      /audio input/i,
      /extracting source audio/i,
    ],
    baseProgress: 5,
    maxProgress: 18,
  },
  {
    id: 'diarization_transcription',
    number: 2,
    title: 'Speaker Diarization & Transcription',
    subtitle: 'Identify & Transcribe',
    description: 'Identifying speakers (who and when) and converting speech to text with precise timestamps.',
    icon: 'record_voice_over',
    patterns: [
      /diarize_and_transcribe/i,
      /transcribing audio/i,
      /diarizing/i,
      /whisper model/i,
      /deepgram transcriber/i,
      /assemblyai/i,
      /pyannote/i,
      /speakers found/i,
      /transcription saved/i,
    ],
    baseProgress: 20,
    maxProgress: 42,
  },
  {
    id: 'translation',
    number: 3,
    title: 'Context-Aware Translation',
    subtitle: 'LLM Translation',
    description: 'Translating dialogue using LLM models considering context, speaker persona, timing, and glossary.',
    icon: 'translate',
    patterns: [
      /translating segments/i,
      /finished translation/i,
      /translate_segments/i,
      /llm translation/i,
    ],
    baseProgress: 45,
    maxProgress: 62,
  },
  {
    id: 'emotions',
    number: 4,
    title: 'Emotion & Prosody Analysis',
    subtitle: 'Tone & Pacing',
    description: 'Evaluating emotional tone, pacing, pauses, and intonation for natural-sounding delivery.',
    icon: 'psychology',
    patterns: [
      /analyzing speech emotions/i,
      /analyze_emotions/i,
      /analyzing prosody/i,
    ],
    baseProgress: 65,
    maxProgress: 74,
  },
  {
    id: 'synthesis',
    number: 5,
    title: 'Voice Cloning & Synthesis',
    subtitle: 'Speech Generation (TTS)',
    description: 'Generating new audio for each segment via selected voice neural networks.',
    icon: 'spatial_audio_off',
    patterns: [
      /synthesizing speech/i,
      /synthesize_speech/i,
      /generating audio for/i,
      /cloning voice/i,
      /resynthesizing/i,
    ],
    baseProgress: 75,
    maxProgress: 90,
  },
  {
    id: 'assembly',
    number: 6,
    title: 'Audio Mixing & Final Assembly',
    subtitle: 'Video & Subtitles',
    description: 'Mixing synthesized dubbing with background music, syncing timestamps, rendering subtitles and final video.',
    icon: 'movie_filter',
    patterns: [
      /mixing all speaker tracks/i,
      /grouping segments by speaker/i,
      /combine_final_video/i,
      /combine_audio_with_video/i,
      /adjusting subtitle timestamps/i,
      /rebuilt translated audio/i,
      /dubbing process completed/i,
    ],
    baseProgress: 92,
    maxProgress: 98,
  },
]

export interface JobProgressViewProps {
  client: ApiClient
  jobId: string
  projectName?: string
  onComplete: (jobId: string) => void
  onCancel?: () => void
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export function JobProgressView({
  client,
  jobId,
  projectName = 'Dubbing Project',
  onComplete,
  onCancel,
}: JobProgressViewProps) {
  const [status, setStatus] = useState<string>('queued')
  const [logs, setLogs] = useState<string[]>([])
  const [error, setError] = useState<string>('')
  const [connection, setConnection] = useState<ConnectionState>('connected')
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [showLogs, setShowLogs] = useState(false)
  const [countdown, setCountdown] = useState<number | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const completedTriggeredRef = useRef(false)

  // Timer for elapsed seconds
  useEffect(() => {
    if (status === 'succeeded' || status === 'failed') return
    const timer = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1)
    }, 1000)
    return () => clearInterval(timer)
  }, [status])

  // Subscribe to job events via SSE
  useEffect(() => {
    let active = true

    // Fetch initial status if available
    client
      .get<{ id: string; status: string; files?: Array<{ id: string; name: string }> }>(
        `/api/jobs/${encodeURIComponent(jobId)}`,
      )
      .then((job) => {
        if (active && job?.status) {
          setStatus((prev) => (prev === 'queued' ? job.status : prev))
        }
      })
      .catch(() => {
        /* ignore, SSE will stream status */
      })

    const unsubscribe = client.subscribeJobEvents(
      jobId,
      (event: SseEvent) => {
        if (!active) return

        if (event.type === 'log') {
          const msg = event.data?.message
          if (typeof msg === 'string') {
            setLogs((prev) => [...prev, msg])
          }
        } else if (event.type === 'state') {
          const nextStatus = event.data?.status
          if (typeof nextStatus === 'string') {
            setStatus(nextStatus)
          }
        } else if (event.type === 'error') {
          const msg = event.data?.message
          if (typeof msg === 'string') {
            setError(msg)
            setStatus('failed')
          }
        } else if (event.type === 'snapshot') {
          const nextStatus = event.data?.status
          if (typeof nextStatus === 'string') {
            setStatus(nextStatus)
          }
          const tail = event.data?.log_tail
          if (Array.isArray(tail)) {
            const newLines = tail
              .map((item) =>
                typeof item === 'object' && item !== null && 'message' in item && typeof item.message === 'string'
                  ? item.message
                  : '',
              )
              .filter(Boolean)
            setLogs((prev) => (prev.length === 0 ? newLines : prev))
          }
          const snapshotError = event.data?.error
          if (
            typeof snapshotError === 'object' &&
            snapshotError !== null &&
            'message' in snapshotError &&
            typeof snapshotError.message === 'string'
          ) {
            setError(snapshotError.message)
            setStatus('failed')
          }
        }
      },
      {
        onState: (st) => {
          if (active) setConnection(st)
        },
        onError: (_err) => {
          if (active) {
            setConnection('reconnecting')
          }
        },
      },
    )

    return () => {
      active = false
      unsubscribe()
    }
  }, [client, jobId])

  // Scroll logs to bottom when expanded
  useEffect(() => {
    if (showLogs && logsEndRef.current) {
      logsEndRef.current.scrollIntoView?.({ behavior: 'smooth' })
    }
  }, [logs, showLogs])

  // Infer current active stage and progress from logs & status
  const { currentStageIndex, calculatedProgress, currentActionText } = useMemo(() => {
    if (status === 'succeeded') {
      return {
        currentStageIndex: PIPELINE_STAGES.length - 1,
        calculatedProgress: 100,
        currentActionText: 'Processing successfully completed! All materials are ready in Studio.',
      }
    }

    if (status === 'failed') {
      return {
        currentStageIndex: 0,
        calculatedProgress: 0,
        currentActionText: error || 'An error occurred during pipeline execution.',
      }
    }

    let highestStageIdx = 0
    let lastMatchedAction = 'Initializing project and preparing environment...'

    // Scan logs backwards or forward to find the highest active stage
    for (const log of logs) {
      for (let sIdx = 0; sIdx < PIPELINE_STAGES.length; sIdx++) {
        const stage = PIPELINE_STAGES[sIdx]
        const matched = stage.patterns.some((p) => p.test(log))
        if (matched) {
          if (sIdx >= highestStageIdx) {
            highestStageIdx = sIdx
            // Clean timestamp or prefix from log line if present
            lastMatchedAction = log.replace(/^(\d{2}:\d{2}:\d{2}\s*-\s*)/, '').trim()
          }
        }
      }
    }

    if (logs.length > 0 && highestStageIdx === 0) {
      const lastLine = logs[logs.length - 1].replace(/^(\d{2}:\d{2}:\d{2}\s*-\s*)/, '').trim()
      if (lastLine) lastMatchedAction = lastLine
    }

    // Determine estimated progress
    const activeStage = PIPELINE_STAGES[highestStageIdx]
    let progress = activeStage.baseProgress
    if (status === 'running') {
      // Progress slightly grows as time or logs increase in stage
      const stageLogCount = logs.length
      const bonus = Math.min(activeStage.maxProgress - activeStage.baseProgress, Math.floor(stageLogCount * 0.8))
      progress = Math.min(98, activeStage.baseProgress + bonus)
    } else if (status === 'queued') {
      progress = 3
      lastMatchedAction = 'Job is queued for execution...'
    }

    return {
      currentStageIndex: highestStageIdx,
      calculatedProgress: progress,
      currentActionText: lastMatchedAction,
    }
  }, [logs, status, error])

  // Handle successful completion auto-redirect countdown
  useEffect(() => {
    if (status === 'succeeded' && !completedTriggeredRef.current) {
      setCountdown(2)
      const interval = setInterval(() => {
        setCountdown((c) => {
          if (c === null || c <= 1) {
            clearInterval(interval)
            if (!completedTriggeredRef.current) {
              completedTriggeredRef.current = true
              onComplete(jobId)
            }
            return 0
          }
          return c - 1
        })
      }, 1000)
      return () => clearInterval(interval)
    }
  }, [status, jobId, onComplete])

  const handleManualProceedToStudio = () => {
    if (!completedTriggeredRef.current) {
      completedTriggeredRef.current = true
      onComplete(jobId)
    }
  }

  const activeStage = PIPELINE_STAGES[currentStageIndex]

  return (
    <section className="job-progress-screen" aria-label="AI Dubbing Progress">
      {/* Top Banner & Header */}
      <div className="job-progress-header">
        <div className="job-progress-header-info">
          <div className="job-progress-tag">
            <span className="material-symbols-outlined spin-anim-slow">motion_mode</span>
            <span>AI DUBBING PIPELINE</span>
          </div>
          <h1 className="job-progress-title">{projectName}</h1>
          <div className="job-progress-meta">
            <span className="meta-badge meta-badge--id">
              <span className="material-symbols-outlined">fingerprint</span>
              ID: {jobId}
            </span>
            <span className={`meta-badge meta-badge--status meta-badge--${status}`}>
              <span className="status-dot"></span>
              {status === 'queued' && 'Queued'}
              {status === 'running' && 'Running'}
              {status === 'succeeded' && 'Succeeded'}
              {status === 'failed' && 'Failed'}
            </span>
            <span className="meta-badge meta-badge--time">
              <span className="material-symbols-outlined">schedule</span>
              Time: {formatElapsed(elapsedSeconds)}
            </span>
            {connection === 'reconnecting' && (
              <span className="meta-badge meta-badge--warn">
                <span className="material-symbols-outlined">sync</span>
                Reconnecting...
              </span>
            )}
          </div>
        </div>

        <div className="job-progress-header-actions">
          {status === 'succeeded' ? (
            <button
              type="button"
              className="proceed-studio-btn"
              onClick={handleManualProceedToStudio}
              aria-label="Go to Studio"
            >
              <span className="material-symbols-outlined">dashboard_customize</span>
              Go to Studio {countdown !== null && countdown > 0 ? `(${countdown}s)` : '→'}
            </button>
          ) : (
            onCancel && (
              <button
                type="button"
                className="cancel-job-btn"
                onClick={onCancel}
                title="Return to Workflow"
              >
                <span className="material-symbols-outlined">arrow_back</span>
                Return
              </button>
            )
          )}
        </div>
      </div>

      {/* Main Large Progress Bar Card */}
      <div className="job-progress-card main-bar-card">
        <div className="progress-bar-top-row">
          <div className="progress-stage-count">
            <span className="stage-num-badge">
              {status === 'succeeded'
                ? 'All stages completed'
                : `Stage ${currentStageIndex + 1} of ${PIPELINE_STAGES.length}`}
            </span>
            <span className="stage-num-title">{activeStage?.title}</span>
          </div>
          <div className="progress-percentage">
            <span className="percentage-number">{calculatedProgress}%</span>
          </div>
        </div>

        {/* The Animated Progress Track */}
        <div className="progress-track" role="progressbar" aria-valuenow={calculatedProgress} aria-valuemin={0} aria-valuemax={100}>
          <div
            className={`progress-fill ${status === 'succeeded' ? 'progress-fill--done' : status === 'failed' ? 'progress-fill--error' : ''}`}
            style={{ width: `${calculatedProgress}%` }}
          >
            <div className="progress-glow"></div>
            <div className="progress-stripes"></div>
          </div>
        </div>

        {/* What is happening right now / Spotlight box */}
        <div className="current-action-box">
          <div className="action-radar">
            {status === 'succeeded' ? (
              <span className="material-symbols-outlined done-icon">check_circle</span>
            ) : status === 'failed' ? (
              <span className="material-symbols-outlined error-icon">error</span>
            ) : (
              <div className="pulse-ring-container">
                <span className="pulse-ring"></span>
                <span className="material-symbols-outlined live-icon">bolt</span>
              </div>
            )}
          </div>
          <div className="action-text-content">
            <div className="action-label">CURRENT ACTION:</div>
            <div className="action-description" title={currentActionText}>
              {currentActionText}
            </div>
          </div>
        </div>
      </div>

      {/* Grid: Stages Overview & Detailed Breakdown */}
      <div className="job-progress-content-grid">
        {/* Left Column: Stages Timeline / Stepper */}
        <div className="job-stages-column">
          <h2 className="section-subtitle">
            <span className="material-symbols-outlined">layers</span>
            Pipeline Stages
          </h2>

          <div className="stages-stepper" role="list">
            {PIPELINE_STAGES.map((stage, idx) => {
              const isPassed = status === 'succeeded' || idx < currentStageIndex
              const isCurrent = status !== 'succeeded' && status !== 'failed' && idx === currentStageIndex
              const isFailed = status === 'failed' && idx === currentStageIndex
              const isUpcoming = !isPassed && !isCurrent && !isFailed

              let stageStatusClass = 'stage--upcoming'
              if (isPassed) stageStatusClass = 'stage--completed'
              else if (isCurrent) stageStatusClass = 'stage--current'
              else if (isFailed) stageStatusClass = 'stage--error'

              return (
                <div
                  key={stage.id}
                  className={`stage-card ${stageStatusClass}`}
                  role="listitem"
                >
                  <div className="stage-card-icon-wrap">
                    {isPassed ? (
                      <span className="material-symbols-outlined stage-status-icon stage-status-icon--check">
                        check
                      </span>
                    ) : isFailed ? (
                      <span className="material-symbols-outlined stage-status-icon stage-status-icon--error">
                        close
                      </span>
                    ) : isCurrent ? (
                      <span className="material-symbols-outlined stage-status-icon stage-status-icon--active spin-anim">
                        sync
                      </span>
                    ) : (
                      <span className="stage-index-num">{stage.number}</span>
                    )}
                  </div>

                  <div className="stage-card-body">
                    <div className="stage-card-header">
                      <div className="stage-title-wrap">
                        <span className="stage-card-title">{stage.title}</span>
                        <span className="stage-card-subtitle">{stage.subtitle}</span>
                      </div>
                      <div className="stage-badge">
                        {isPassed && <span className="badge badge--success">Completed</span>}
                        {isCurrent && <span className="badge badge--active">Running...</span>}
                        {isFailed && <span className="badge badge--danger">Error</span>}
                        {isUpcoming && <span className="badge badge--pending">Queued</span>}
                      </div>
                    </div>

                    <p className="stage-card-description">{stage.description}</p>

                    {isCurrent && (
                      <div className="stage-live-indicator">
                        <span className="live-dot"></span>
                        <span className="live-msg">Active process...</span>
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Right Column: Live Terminal / Logs Drawer */}
        <div className="job-terminal-column">
          <div className="terminal-header">
            <div className="terminal-title">
              <span className="material-symbols-outlined">terminal</span>
              <span>Live Execution Logs</span>
              <span className="log-count-tag">{logs.length} messages</span>
            </div>
            <button
              type="button"
              className="toggle-logs-btn"
              onClick={() => setShowLogs((v) => !v)}
              aria-label="Collapse / Expand Logs"
            >
              <span className="material-symbols-outlined">
                {showLogs ? 'unfold_less' : 'unfold_more'}
              </span>
              {showLogs ? 'Collapse' : 'Expand'}
            </button>
          </div>

          <div className={`terminal-console ${showLogs ? 'terminal-console--expanded' : ''}`}>
            {logs.length === 0 ? (
              <div className="terminal-empty">
                <span className="material-symbols-outlined spin-anim">hourglass_top</span>
                <span>Waiting for server output...</span>
              </div>
            ) : (
              <div className="terminal-lines">
                {logs.map((log, index) => (
                  <div key={`${index}-${log}`} className="terminal-line">
                    <span className="line-num">{index + 1}</span>
                    <span className="line-text">{log}</span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            )}
          </div>

          {/* Success Banner when finished */}
          {status === 'succeeded' && (
            <div className="completion-card">
              <div className="completion-icon">
                <span className="material-symbols-outlined">task_alt</span>
              </div>
              <div className="completion-content">
                <h3>Dubbing successfully generated!</h3>
                <p>
                  All audio tracks, subtitles and video are ready. Automatically redirecting to{' '}
                  <strong>HukFlow Studio</strong> for editing and export.
                </p>
                <button
                  type="button"
                  className="completion-jump-btn"
                  onClick={handleManualProceedToStudio}
                >
                  Open in Studio now →
                </button>
              </div>
            </div>
          )}

          {/* Error Banner when failed */}
          {status === 'failed' && (
            <div className="error-card" role="alert">
              <div className="error-icon">
                <span className="material-symbols-outlined">error</span>
              </div>
              <div className="error-content">
                <h3>Dubbing error</h3>
                <p>{error || 'An error occurred while processing the video. Check the logs above.'}</p>
                {onCancel && (
                  <button type="button" className="error-retry-btn" onClick={onCancel}>
                    <span className="material-symbols-outlined">refresh</span>
                    Return to Workflow and retry
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
