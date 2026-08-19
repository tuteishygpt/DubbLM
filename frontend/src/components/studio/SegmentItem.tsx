import { formatTime, parseTimeInput } from '../../utils/time'
import type { Segment } from '../../types/models'

const SPEAKER_COLOURS = ['speaker-0', 'speaker-1', 'speaker-2', 'speaker-3'] as const

import { useStudio } from '../../contexts/StudioContext'

export interface SegmentItemProps {
  segment: Segment
  isActive: boolean
  speakerIdx: number
  isSourceExpanded: boolean
  segmentIndex: number
  totalSegments: number
}

export function SegmentItem({
  segment: seg,
  isActive,
  speakerIdx,
  isSourceExpanded,
  segmentIndex,
  totalSegments,
}: SegmentItemProps) {
  const {
    displaySpeakers,
    playingAudioSegmentId,
    regeneratingId,
    handleSelectSegment: onSelect,
    handleSpeakerChange: onSpeakerChange,
    handleTimestampChange: onTimestampChange,
    handleSourceTextChange: onSourceTextChange,
    handleTranslationChange: onTranslationChange,
    toggleSource: onToggleSource,
    handlePlaySegmentAudio: onPlayAudio,
    handleRegenerateAudio: onRegenerateAudio,
    handleDeleteSegment: onDelete,
  } = useStudio()

  const hasAudio = seg.audio !== null
  const speakerSlot = SPEAKER_COLOURS[speakerIdx % 4]

  // Calculate dynamic row count based on text length and newlines
  const explicitLines = seg.translation.split('\n').length
  const estimatedLines = Math.ceil((seg.translation.length || 1) / 75)
  const dynamicRows = Math.max(1, Math.min(8, Math.max(explicitLines, estimatedLines)))

  return (
    <div
      id={`transcript-card-${seg.segment_id}`}
      onClick={() => onSelect(seg)}
      className={`transcript-card ${isActive ? 'active-card' : 'inactive-card'}`}
      data-testid={`transcript-card-${seg.segment_id}`}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onSelect(seg)}
    >
      <div className={`card-accent-line ${speakerSlot}`}></div>

      <div className="card-header">
        <div className="speaker-info">
          <span className={`speaker-dot speaker-dot-${speakerIdx % 4}`}></span>
          <select
            className="speaker-name-select"
            value={seg.speaker}
            onChange={(e) => { e.stopPropagation(); onSpeakerChange(seg.segment_id, e.target.value) }}
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
                if (v !== null) onTimestampChange(seg.segment_id, 'start', v)
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
                if (v !== null) onTimestampChange(seg.segment_id, 'end', v)
                else e.target.value = formatTime(seg.end)
              }}
              onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
              title="End time (MM:SS)"
            />
          </span>
          {totalSegments > 1 && (
            <span className="segment-stepper-label">
              ({segmentIndex + 1}/{totalSegments})
            </span>
          )}
          <span className="dubbed-tag">DUBBED (BE)</span>
          <span className="text-count-badge">{seg.translation.length} chars</span>
        </div>

        <div className="card-status-row">
          <button
            type="button"
            className={`source-toggle-btn ${isSourceExpanded ? 'active' : ''}`}
            onClick={(e) => {
              e.stopPropagation()
              onToggleSource(seg.segment_id)
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
                onPlayAudio(seg.segment_id, seg.audio!.url)
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
              void onRegenerateAudio(seg.segment_id)
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
              onDelete(seg.segment_id)
            }}
            title={`Delete segment ${seg.speaker}`}
            data-testid={`delete-btn-${seg.segment_id}`}
            aria-label={`Delete segment ${seg.speaker}`}
          >
            <span className="material-symbols-outlined">delete</span>
          </button>
        </div>
      </div>

      {isSourceExpanded && (
        <div className="source-drawer" onClick={(e) => e.stopPropagation()}>
          <div className="source-drawer-header">
            <span className="col-label">Source ({seg.speaker})</span>
            <button
              type="button"
              className="icon-btn close-source-btn"
              onClick={() => onToggleSource(seg.segment_id)}
              title="Hide source text"
            >
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>
          <textarea
            className={`source-text-textarea source-text-${speakerSlot.replace('speaker-', '')}`}
            value={seg.text}
            onChange={(e) => onSourceTextChange(seg.segment_id, e.target.value)}
            rows={2}
            aria-label={`Source text for ${seg.speaker}`}
          />
        </div>
      )}

      <div className="editable-dubbed-wrapper">
        <textarea
          className="dubbed-textarea"
          value={seg.synthesized_text || seg.translation}
          onChange={(e) => onTranslationChange(seg.segment_id, e.target.value)}
          onClick={(e) => {
            e.stopPropagation()
            onSelect(seg)
          }}
          onFocus={() => onSelect(seg)}
          rows={dynamicRows}
          aria-label={`Dubbed text ${seg.speaker}`}
          placeholder="Dubbed text in Belarusian..."
        />
      </div>
    </div>
  )
}
