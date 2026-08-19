import { useEffect, useRef } from 'react'
import type { Segment } from '../../types/models'
import { formatTime } from '../../utils/time'
import { SegmentWaveform } from './SegmentWaveform'
import { useStudio } from '../../contexts/StudioContext'

export function TimelineCanvas() {
  const {
    displaySpeakers,
    displaySegments,
    currentActive,
    musicFiles,
    backgroundDurations,
    audioDurations,
    mutedTracks,
    toggleTrackMute: onToggleTrackMute,
    zoomLevel,
    currentTime,
    duration,
    isPlaying,
    seekToTime: onSeek,
    handleSelectSegment: onSelectSegment,
  } = useStudio()

  const timelineCanvasRef = useRef<HTMLDivElement>(null)

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

  return (
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
              onClick={() => onToggleTrackMute(speaker)}
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
          <button type="button" className="mute-btn" onClick={() => onToggleTrackMute('a1')} title="Toggle Mute">
            <span className="material-symbols-outlined">{mutedTracks.a1 ? 'volume_off' : 'volume_up'}</span>
          </button>
        </div>
      </div>

      {/* Timeline canvas with horizontal scroll & zoom */}
      <div ref={timelineCanvasRef} className="timeline-tracks-canvas">
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
            onSeek(pct * duration)
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
                      onClick={() => onSelectSegment(seg)}
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
  )
}
