import { useWaveform } from '../../hooks/useWaveform'

export interface SegmentWaveformProps {
  url: string | null | undefined
  /** SVG viewBox width (bars count) */
  bars?: number
}

/** Renders a real audio waveform — discrete rounded vertical bars, mirrored from centre */
export function SegmentWaveform({ url, bars = 80 }: SegmentWaveformProps) {
  const peaks = useWaveform(url, bars)

  // viewBox: each bar occupies 3 units (1 bar + 2 gap) → thinner look
  const vbW = bars * 3
  const vbH = 20
  const cx  = vbH / 2   // 10 — vertical centre
  const maxH = 8.5       // max half-height

  if (peaks.length === 0) {
    // Placeholder: row of tiny dim stubs while audio is loading
    return (
      <svg className="waveform-svg" viewBox={`0 0 ${vbW} ${vbH}`} preserveAspectRatio="none">
        {Array.from({ length: bars }, (_, i) => (
          <line
            key={i}
            x1={i * 3 + 1.5} y1={cx - 0.8}
            x2={i * 3 + 1.5} y2={cx + 0.8}
            stroke="currentColor" strokeWidth="0.5" strokeLinecap="round"
            opacity="0.25"
          />
        ))}
      </svg>
    )
  }

  return (
    <svg className="waveform-svg" viewBox={`0 0 ${vbW} ${vbH}`} preserveAspectRatio="none">
      {peaks.map((amp, i) => {
        const h = Math.max(0.5, amp * maxH)
        const x = i * 3 + 1.5   // centre x of bar (1 unit wide, 2 units gap)
        return (
          <line
            key={i}
            x1={x} y1={cx - h}
            x2={x} y2={cx + h}
            stroke="currentColor"
            strokeWidth="0.5"
            strokeLinecap="round"
            opacity="0.85"
          />
        )
      })}
    </svg>
  )
}
