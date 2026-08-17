import { useEffect, useRef, useState } from 'react'
import type { ApiClient } from '../api/types'

export interface ExportFileItem {
  id: string
  name: string
  kind: string
  size?: number
  url?: string
}

export interface ExportDropdownProps {
  client?: ApiClient
  activeJobId?: string | null
  activeProjectName?: string | null
  videoSrc?: string | null
  audioSrc?: string | null
  files?: ExportFileItem[]
  disabled?: boolean
}

/**
 * Downloads a file via Blob / ObjectURL or fallback anchor click.
 */
export async function triggerFileDownload(url: string, filename: string): Promise<void> {
  try {
    const res = await fetch(url)
    if (!res.ok) {
      throw new Error(`Download failed with status ${res.status}`)
    }
    const blob = await res.blob()
    const blobUrl = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = blobUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    setTimeout(() => window.URL.revokeObjectURL(blobUrl), 15000)
  } catch {
    // Direct anchor fallback
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    link.target = '_blank'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }
}

export function ExportDropdown({
  client,
  activeJobId,
  activeProjectName,
  videoSrc,
  audioSrc,
  files,
  disabled = false,
}: ExportDropdownProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [exportingType, setExportingType] = useState<'video' | 'audio' | null>(null)
  const [feedbackMessage, setFeedbackMessage] = useState<{ text: string; isError: boolean } | null>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Close dropdown on click outside
  useEffect(() => {
    if (!isOpen) return
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [isOpen])

  // Helper to fetch files for a job if not already passed in props
  const resolveJobFiles = async (jobId: string): Promise<ExportFileItem[]> => {
    if (files && files.length > 0) return files
    if (!client) return []
    try {
      const res = await client.get<{ files?: ExportFileItem[] }>(`/api/jobs/${encodeURIComponent(jobId)}/files`)
      return Array.isArray(res?.files) ? res.files : []
    } catch {
      return []
    }
  }

  // Handle Video Export
  const handleExportVideo = async () => {
    setIsOpen(false)
    setExportingType('video')
    setFeedbackMessage(null)

    try {
      let targetUrl: string | null = null
      let targetFilename = `${activeProjectName ? activeProjectName.trim().replace(/\s+/g, '_') : 'dubbed'}_video.mp4`

      // 1. Try to get registered video from active job files
      if (activeJobId) {
        const jobFiles = await resolveJobFiles(activeJobId)
        const videoFile = jobFiles.find(
          (f) =>
            f.kind === 'output_video' ||
            f.kind === 'result' ||
            f.kind === 'video' ||
            /\.(mp4|webm|mov|mkv)$/i.test(f.name),
        )
        if (videoFile) {
          targetUrl = videoFile.url || `/api/jobs/${encodeURIComponent(activeJobId)}/files/${encodeURIComponent(videoFile.id)}`
          targetFilename = videoFile.name || targetFilename
        }
      }

      // 2. If no job file found, fallback to videoSrc if valid
      if (!targetUrl && videoSrc && videoSrc.length > 0) {
        targetUrl = videoSrc
        if (videoSrc.startsWith('http') || videoSrc.startsWith('/')) {
          const urlParts = videoSrc.split('/')
          const lastPart = urlParts[urlParts.length - 1].split('?')[0]
          if (/\.(mp4|webm|mov|mkv)$/i.test(lastPart)) {
            targetFilename = lastPart
          }
        }
      }

      // 3. Fallback: check if client can find any ready project / job
      if (!targetUrl && client) {
        try {
          const jobsRes = await client.get<{ jobs?: Array<{ id: string; status: string; files?: ExportFileItem[] }> }>('/api/jobs')
          const succeededJob = jobsRes?.jobs?.find((j) => (j.status === 'succeeded' || j.status === 'running') && j.files && j.files.length > 0)
          if (succeededJob && succeededJob.files) {
            const vid = succeededJob.files.find((f) => f.kind === 'output_video' || f.kind === 'video' || /\.(mp4|webm)$/i.test(f.name))
            if (vid) {
              targetUrl = vid.url || `/api/jobs/${encodeURIComponent(succeededJob.id)}/files/${encodeURIComponent(vid.id)}`
              targetFilename = vid.name || targetFilename
            }
          }
        } catch {
          // ignore
        }
      }

      // 4. Default demo video fallback if nothing else found
      if (!targetUrl) {
        targetUrl = '/video.mp4'
      }

      await triggerFileDownload(targetUrl, targetFilename)
      setFeedbackMessage({ text: `Video download started (${targetFilename})`, isError: false })
      setTimeout(() => setFeedbackMessage(null), 4000)
    } catch (err) {
      setFeedbackMessage({
        text: err instanceof Error ? err.message : 'Failed to export video',
        isError: true,
      })
      setTimeout(() => setFeedbackMessage(null), 5000)
    } finally {
      setExportingType(null)
    }
  }

  // Handle Audio Export
  const handleExportAudio = async () => {
    setIsOpen(false)
    setExportingType('audio')
    setFeedbackMessage(null)

    try {
      let targetUrl: string | null = null
      let targetFilename = `${activeProjectName ? activeProjectName.trim().replace(/\s+/g, '_') : 'dubbed'}_audio.wav`

      // 1. Try to get registered audio from active job files
      if (activeJobId) {
        const jobFiles = await resolveJobFiles(activeJobId)
        // Prioritize output audio, then background/music, then vocal/source
        const audioFile =
          jobFiles.find((f) => f.kind === 'output_audio' || f.name.toLowerCase().includes('output.wav')) ||
          jobFiles.find((f) => f.kind === 'background_audio' || f.kind === 'music') ||
          jobFiles.find((f) => f.kind === 'audio' || /\.(wav|mp3|m4a|aac|flac)$/i.test(f.name))

        if (audioFile) {
          targetUrl = audioFile.url || `/api/jobs/${encodeURIComponent(activeJobId)}/files/${encodeURIComponent(audioFile.id)}`
          targetFilename = audioFile.name || targetFilename
        }
      }

      // 2. Fallback to audioSrc if provided
      if (!targetUrl && audioSrc && audioSrc.length > 0) {
        targetUrl = audioSrc
        const urlParts = audioSrc.split('/')
        const lastPart = urlParts[urlParts.length - 1].split('?')[0]
        if (/\.(wav|mp3|m4a|aac|flac)$/i.test(lastPart)) {
          targetFilename = lastPart
        }
      }

      // 3. Fallback: check ready projects/jobs via client
      if (!targetUrl && client) {
        try {
          const jobsRes = await client.get<{ jobs?: Array<{ id: string; status: string; files?: ExportFileItem[] }> }>('/api/jobs')
          const succeededJob = jobsRes?.jobs?.find((j) => (j.status === 'succeeded' || j.status === 'running') && j.files && j.files.length > 0)
          if (succeededJob && succeededJob.files) {
            const aud =
              succeededJob.files.find((f) => f.kind === 'output_audio' || f.name.toLowerCase().includes('output.wav')) ||
              succeededJob.files.find((f) => f.kind === 'background_audio' || f.kind === 'audio' || /\.(wav|mp3)$/i.test(f.name))
            if (aud) {
              targetUrl = aud.url || `/api/jobs/${encodeURIComponent(succeededJob.id)}/files/${encodeURIComponent(aud.id)}`
              targetFilename = aud.name || targetFilename
            }
          }
        } catch {
          // ignore
        }
      }

      if (!targetUrl) {
        setFeedbackMessage({ text: 'No audio file found for export.', isError: true })
        setTimeout(() => setFeedbackMessage(null), 4000)
        return
      }

      await triggerFileDownload(targetUrl, targetFilename)
      setFeedbackMessage({ text: `Audio download started (${targetFilename})`, isError: false })
      setTimeout(() => setFeedbackMessage(null), 4000)
    } catch (err) {
      setFeedbackMessage({
        text: err instanceof Error ? err.message : 'Failed to export audio',
        isError: true,
      })
      setTimeout(() => setFeedbackMessage(null), 5000)
    } finally {
      setExportingType(null)
    }
  }

  const hasProject = Boolean(activeJobId || activeProjectName)
  const isButtonDisabled = disabled || !hasProject || exportingType !== null

  return (
    <div className="export-menu-wrap" ref={dropdownRef}>
      <button
        type="button"
        className="export-btn"
        onClick={() => {
          if (!isButtonDisabled) {
            setIsOpen((prev) => !prev)
          }
        }}
        disabled={isButtonDisabled}
        aria-haspopup="true"
        aria-expanded={isOpen}
        aria-label="Export"
        title={
          !hasProject
            ? 'No project selected for export'
            : exportingType
            ? 'Export in progress…'
            : 'Export Video or Audio'
        }
      >
        {exportingType ? (
          <span className="material-symbols-outlined spin-anim" style={{ fontSize: '14px', marginRight: '4px' }}>
            sync
          </span>
        ) : (
          <span className="material-symbols-outlined" style={{ fontSize: '14px', marginRight: '4px' }}>
            download
          </span>
        )}
        <span>{exportingType ? 'Exporting…' : 'Export'}</span>
        <span className="material-symbols-outlined" style={{ fontSize: '14px', marginLeft: '4px' }}>
          {isOpen ? 'expand_less' : 'expand_more'}
        </span>
      </button>

      {isOpen && !isButtonDisabled && (
        <div className="export-dropdown-menu" role="menu" aria-label="Export options">
          <div className="export-menu-header">
            <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>
              ios_share
            </span>
            <span>Export Media</span>
          </div>

          {/* Option 1: Video */}
          <button
            type="button"
            className="export-dropdown-item"
            role="menuitem"
            onClick={handleExportVideo}
            data-testid="export-video-btn"
          >
            <div className="export-item-icon export-item-icon--video">
              <span className="material-symbols-outlined">movie</span>
            </div>
            <div className="export-item-body">
              <div className="export-item-title-row">
                <span className="export-item-title">Video</span>
                <span className="export-item-badge">.MP4</span>
              </div>
              <span className="export-item-desc">Full localized dubbed video</span>
            </div>
          </button>

          {/* Option 2: Audio */}
          <button
            type="button"
            className="export-dropdown-item"
            role="menuitem"
            onClick={handleExportAudio}
            data-testid="export-audio-btn"
          >
            <div className="export-item-icon export-item-icon--audio">
              <span className="material-symbols-outlined">audiotrack</span>
            </div>
            <div className="export-item-body">
              <div className="export-item-title-row">
                <span className="export-item-title">Audio</span>
                <span className="export-item-badge">.WAV</span>
              </div>
              <span className="export-item-desc">Master dubbed & mixed audio</span>
            </div>
          </button>
        </div>
      )}

      {/* Floating feedback toast */}
      {feedbackMessage && (
        <div
          className={`export-feedback-toast ${feedbackMessage.isError ? 'export-feedback-toast--error' : 'export-feedback-toast--success'}`}
          role="status"
        >
          <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>
            {feedbackMessage.isError ? 'error' : 'check_circle'}
          </span>
          <span>{feedbackMessage.text}</span>
        </div>
      )}
    </div>
  )
}
