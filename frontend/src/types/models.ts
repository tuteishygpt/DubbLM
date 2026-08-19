import { OptionsResponse } from '../api/types'

export interface Job {
  id: string
  status: string
  project_name?: string | null
  files: JobFile[]
}

export interface VoiceProfile {
  tts_system: string
  model?: string
  voice_name?: string
  style_prompt?: string
  reference_mode?: string
  reference_audio?: string
  reference_text?: string
  params?: Record<string, unknown>
}

export interface ProfilesSnapshot {
  revision: string
  profiles: Record<string, VoiceProfile>
}

export interface ReferenceEntry {
  speaker_id: string
  reference_text: string
  audio: { id: string; name: string; url: string }
}

export interface ReferencesSnapshot {
  revision: string
  entries: ReferenceEntry[]
}

export interface ProjectSummary {
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

export interface JobFile {
  id: string
  name: string
  kind: string
  size: number
  url?: string
}

export interface SegmentAudio {
  id: string
  name: string
  url: string
}

export interface Segment {
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

export interface TextDocument {
  revision: string
  source: string
  segments: Segment[]
}

export type ExtendedOptions = OptionsResponse & {
  tts_providers?: string[]
  tts_models?: Record<string, string[]>
  tts_voices?: Record<string, string[]>
  tts_reference_capabilities?: Record<string, string>
  reference_modes?: string[]
}
