import { RefObject } from 'react'
import { formatTimecode } from '../../utils/time'
import type { Segment } from '../../types/models'

export interface VideoPlayerCanvasProps {
  videoRef: RefObject<HTMLVideoElement>
  videoSrc: string
  currentTime: number
  setCurrentTime: (time: number) => void
  duration: number
  videoDuration: number
  setVideoDuration: (time: number) => void
  isPlaying: boolean
  setIsPlaying: (playing: boolean) => void
  togglePlayPause: () => void
  liveSubSegment: Segment | undefined
  handleScrubberClick: (e: React.MouseEvent<HTMLDivElement>) => void
  displaySegments: Segment[]
  currentActive: Segment | null
  handleSelectSegment: (seg: Segment) => void
}

export function VideoPlayerCanvas({
  videoRef,
  videoSrc,
  currentTime,
  setCurrentTime,
  duration,
  videoDuration,
  setVideoDuration,
  isPlaying,
  setIsPlaying,
  togglePlayPause,
  liveSubSegment,
  handleScrubberClick,
  displaySegments,
  currentActive,
  handleSelectSegment,
}: VideoPlayerCanvasProps) {
  return (
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
  )
}
