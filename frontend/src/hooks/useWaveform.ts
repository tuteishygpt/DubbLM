import { useState, useEffect } from 'react'

/** Cache decoded waveform bar arrays by URL so we only fetch + decode each file once */
const waveformCache = new Map<string, number[]>()

/**
 * Decodes an audio URL via Web Audio API and returns an array of `bars` RMS
 * amplitude values (0‥1) suitable for SVG rendering.
 */
export function useWaveform(url: string | null | undefined, bars = 80): number[] {
  const [peaks, setPeaks] = useState<number[]>([])

  useEffect(() => {
    if (!url) { setPeaks([]); return }

    // Return cached result immediately if available
    const cached = waveformCache.get(url)
    if (cached) { setPeaks(cached); return }

    let cancelled = false
    const AudioContextClass =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioContextClass) {
      return
    }
    const audioCtx = new AudioContextClass()

    fetch(url)
      .then((r) => r.arrayBuffer())
      .then((buf) => audioCtx.decodeAudioData(buf))
      .then((decoded) => {
        if (cancelled) return
        // Mix all channels into mono
        const channelData = decoded.getChannelData(0)
        const blockSize = Math.floor(channelData.length / bars)
        const result: number[] = []
        for (let i = 0; i < bars; i++) {
          const offset = i * blockSize
          let sum = 0
          for (let j = 0; j < blockSize; j++) {
            sum += channelData[offset + j] ** 2
          }
          result.push(Math.sqrt(sum / blockSize))
        }
        // Normalise to 0‥1
        const max = Math.max(...result, 1e-6)
        const normalised = result.map((v) => v / max)
        waveformCache.set(url, normalised)
        if (!cancelled) setPeaks(normalised)
      })
      .catch(() => { if (!cancelled) setPeaks([]) })
      .finally(() => { void audioCtx.close() })

    return () => { cancelled = true }
  }, [url, bars])

  return peaks
}
