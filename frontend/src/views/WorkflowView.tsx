import { FormEvent, useEffect, useState } from 'react'
import type { ApiClient, ConfigResponse, JsonValue, OptionsResponse, SchemaFieldDefinition, SelectOption } from '../api/types'

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

interface ProfilesSnapshot { revision: string; profiles: Record<string, VoiceProfile> }
interface ReferenceEntry { speaker_id: string; reference_text: string; audio: { id: string; name: string; url: string } }
interface ReferencesSnapshot { revision: string; entries: ReferenceEntry[] }

type ExtendedOptions = OptionsResponse & {
  tts_providers?: string[]
  tts_models?: Record<string, string[]>
  tts_voices?: Record<string, string[]>
  tts_reference_capabilities?: Record<string, string>
  reference_modes?: string[]
}

const DEFAULT_PROFILE: VoiceProfile = {
  tts_system: 'gemini',
  model: 'gemini-2.5-pro-preview-tts',
  voice_name: 'Kore',
  style_prompt: '',
  reference_mode: 'none',
  reference_audio: '',
}

const READY_PRESETS: Array<{ label: string; profile: Partial<VoiceProfile> }> = [
  { label: 'Gemini - Kore (Clear narrator)', profile: { tts_system: 'gemini', model: 'gemini-2.5-pro-preview-tts', voice_name: 'Kore', style_prompt: 'clear narrator', reference_mode: 'none' } },
  { label: 'Gemini - Puck (Friendly & natural)', profile: { tts_system: 'gemini', model: 'gemini-2.5-pro-preview-tts', voice_name: 'Puck', style_prompt: 'friendly, natural', reference_mode: 'none' } },
  { label: 'Gemini - Zephyr (Calm & soft)', profile: { tts_system: 'gemini', model: 'gemini-2.5-pro-preview-tts', voice_name: 'Zephyr', style_prompt: 'calm, soft', reference_mode: 'none' } },
  { label: 'Gemini - Fenrir (Deep voice)', profile: { tts_system: 'gemini', model: 'gemini-2.5-pro-preview-tts', voice_name: 'Fenrir', style_prompt: 'deep and steady', reference_mode: 'none' } },
  { label: 'OpenAI - Alloy (Neutral & balanced)', profile: { tts_system: 'openai', model: 'tts-1', voice_name: 'alloy', reference_mode: 'none' } },
  { label: 'OpenAI - Nova (Energetic female)', profile: { tts_system: 'openai', model: 'tts-1', voice_name: 'nova', reference_mode: 'none' } },
  { label: 'OpenAI - Onyx (Warm male)', profile: { tts_system: 'openai', model: 'tts-1', voice_name: 'onyx', reference_mode: 'none' } },
  { label: 'OmniVoice - Voice Cloning (auto track)', profile: { tts_system: 'omnivoice', reference_mode: 'speaker' } },
  { label: 'XTTS - Voice Cloning (auto track)', profile: { tts_system: 'xtts', reference_mode: 'speaker' } },
  { label: 'Higgs - Voice Cloning (auto track)', profile: { tts_system: 'higgs', reference_mode: 'speaker' } },
  { label: 'BexTTS - Voice Cloning (auto track)', profile: { tts_system: 'bextts', reference_mode: 'speaker' } },
  { label: 'F5-TTS - Voice Cloning (auto track)', profile: { tts_system: 'f5', reference_mode: 'speaker' } },
]

export function WorkflowView({ client, config }: { client: ApiClient; config: ConfigResponse }) {
  const fields = config.schema.fields.filter((field) => field.workflow || field.scope === 'workflow')
  const regularFields = fields.filter((f) => f.type !== 'boolean')
  const booleanFields = fields.filter((f) => f.type === 'boolean')

  const [options, setOptions] = useState<ExtendedOptions>({})
  const [values, setValues] = useState<Record<string, JsonValue>>(() =>
    Object.fromEntries(fields.map((field) => [field.name, config.values[field.name] ?? '']))
  )
  const [projectName, setProjectName] = useState('Project Alpha')
  const [video, setVideo] = useState<File>()
  const [tracks, setTracks] = useState<File[]>([])
  const [mapping, setMapping] = useState('{}')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [isStarting, setIsStarting] = useState(false)

  // ── Voice settings state ──────────────────────────────────────────────────
  const [savedProfiles, setSavedProfiles] = useState<Record<string, VoiceProfile>>({})
  const [referenceEntries, setReferenceEntries] = useState<ReferenceEntry[]>([])
  const [voiceProfiles, setVoiceProfiles] = useState<Record<string, VoiceProfile>>({
    '*': { ...DEFAULT_PROFILE },
  })
  const [activeSpeakerTab, setActiveSpeakerTab] = useState<string>('*')
  const [customSpeakerInput, setCustomSpeakerInput] = useState('')
  const [enableVoiceOverrides, setEnableVoiceOverrides] = useState(true)
  const [showAdvancedVoice, setShowAdvancedVoice] = useState(false)

  useEffect(() => {
    client.get<ExtendedOptions>('/api/options')
      .then((data) => setOptions(data))
      .catch((reason) => setError(String(reason)))

    client.get<ProfilesSnapshot>('/api/voice-profiles')
      .then((data) => {
        if (data?.profiles) {
          setSavedProfiles(data.profiles)
          if (data.profiles['*']) {
            setVoiceProfiles((prev) => ({ ...prev, '*': { ...data.profiles['*'] } }))
          }
        }
      })
      .catch(() => {/* ignore if unavailable */})

    client.get<ReferencesSnapshot>('/api/reference-library')
      .then((data) => {
        if (Array.isArray(data?.entries)) setReferenceEntries(data.entries)
      })
      .catch(() => {/* ignore if unavailable */})
  }, [client])

  // Parse speaker mapping whenever mapping text changes from outside
  const speakerToFileNameMap: Record<string, string> = (() => {
    try {
      const parsed = JSON.parse(mapping)
      return isStringRecord(parsed) ? parsed : {}
    } catch {
      return {}
    }
  })()

  // Reverse mapping for track UI: fileName -> speaker
  const fileNameToSpeakerMap: Record<string, string> = Object.fromEntries(
    Object.entries(speakerToFileNameMap).map(([spk, fn]) => [fn, spk])
  )

  function updateSpeakerMapping(newMapping: Record<string, string>) {
    setMapping(JSON.stringify(newMapping, null, 2))
    // Automatically ensure speaker tabs exist in voiceProfiles
    setVoiceProfiles((current) => {
      const updated = { ...current }
      for (const spk of Object.keys(newMapping)) {
        if (!updated[spk]) {
          updated[spk] = { ...(updated['*'] ?? DEFAULT_PROFILE) }
        }
      }
      return updated
    })
  }

  function handleAddTracks(newFiles: File[]) {
    const combined = [...tracks]
    const updatedMapping = { ...speakerToFileNameMap }

    newFiles.forEach((file, index) => {
      if (!combined.some((f) => f.name === file.name)) {
        combined.push(file)
        // Deduce default speaker label from filename
        const deduced = deduceSpeakerLabel(file.name, combined.length - 1 + index)
        updatedMapping[deduced] = file.name
      }
    })

    setTracks(combined)
    updateSpeakerMapping(updatedMapping)
  }

  function handleRemoveTrack(fileName: string) {
    const updatedTracks = tracks.filter((t) => t.name !== fileName)
    setTracks(updatedTracks)
    const updatedMapping = { ...speakerToFileNameMap }
    for (const [spk, fn] of Object.entries(updatedMapping)) {
      if (fn === fileName) delete updatedMapping[spk]
    }
    updateSpeakerMapping(updatedMapping)
  }

  function handleTrackSpeakerChange(fileName: string, newSpeaker: string) {
    const cleanSpeaker = newSpeaker.trim()
    if (!cleanSpeaker) return
    const updatedMapping = { ...speakerToFileNameMap }
    // Remove old assignment for this file
    for (const [spk, fn] of Object.entries(updatedMapping)) {
      if (fn === fileName) delete updatedMapping[spk]
    }
    updatedMapping[cleanSpeaker] = fileName
    updateSpeakerMapping(updatedMapping)
  }

  function handleVoiceProfileChange(speaker: string, partial: Partial<VoiceProfile>) {
    setVoiceProfiles((prev) => ({
      ...prev,
      [speaker]: {
        ...(prev[speaker] ?? prev['*'] ?? DEFAULT_PROFILE),
        ...partial,
      },
    }))
  }

  function handleApplySavedProfile(speaker: string, profileKey: string) {
    const profile = savedProfiles[profileKey]
    if (profile) {
      handleVoiceProfileChange(speaker, { ...profile })
    }
  }

  function handleApplyPreset(speaker: string, presetIndex: number) {
    const preset = READY_PRESETS[presetIndex]
    if (preset) {
      handleVoiceProfileChange(speaker, { ...preset.profile })
    }
  }

  function handleAddCustomSpeaker() {
    const clean = customSpeakerInput.trim()
    if (!clean) return
    if (!voiceProfiles[clean]) {
      setVoiceProfiles((prev) => ({
        ...prev,
        [clean]: { ...(prev['*'] ?? DEFAULT_PROFILE) },
      }))
    }
    setActiveSpeakerTab(clean)
    setCustomSpeakerInput('')
  }

  function handleRemoveCustomSpeaker(speaker: string) {
    if (speaker === '*') return
    setVoiceProfiles((prev) => {
      const next = { ...prev }
      delete next[speaker]
      return next
    })
    if (activeSpeakerTab === speaker) setActiveSpeakerTab('*')
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (isStarting) return
    setError('')
    if (!video) { setError('Video is required'); return }

    let speakerLabelMapping: Record<string, string>
    setIsStarting(true)
    try {
      const parsed: unknown = JSON.parse(mapping)
      if (!isStringRecord(parsed)) throw new Error()
      speakerLabelMapping = parsed
    } catch {
      setError('Speaker-label mapping must be a JSON object of speaker IDs to file names')
      return
    }

    try {
      const videoUpload = await upload(client, video)
      const uploadedTracks = new Map<string, string>()
      for (const track of tracks) {
        uploadedTracks.set(track.name, (await upload(client, track)).id)
      }

      const isolatedTracks = Object.fromEntries(
        Object.entries(speakerLabelMapping).map(([speaker, fileName]) => {
          const uploadId = uploadedTracks.get(fileName)
          if (!uploadId) throw new Error(`No isolated track named ${fileName}`)
          return [speaker, uploadId]
        })
      )

      const overrides: Record<string, JsonValue> = {
        project_name: projectName.trim() || 'Untitled project',
        ...Object.fromEntries(fields.map((field) => [field.name, values[field.name]])),
      }

      // Attach voice profiles to overrides if configured
      if (enableVoiceOverrides && Object.keys(voiceProfiles).length > 0) {
        // Clean empty values
        const cleanVoices: Record<string, Record<string, unknown>> = {}
        for (const [spk, prof] of Object.entries(voiceProfiles)) {
          const p: Record<string, unknown> = { tts_system: prof.tts_system }
          if (prof.model) p.model = prof.model
          if (prof.voice_name) p.voice_name = prof.voice_name
          if (prof.style_prompt) p.style_prompt = prof.style_prompt
          if (prof.reference_mode && prof.reference_mode !== 'none') p.reference_mode = prof.reference_mode
          if (prof.reference_audio) p.reference_audio = prof.reference_audio
          if (prof.reference_text) p.reference_text = prof.reference_text
          if (prof.params && Object.keys(prof.params).length > 0) p.params = prof.params
          cleanVoices[spk] = p
        }
        overrides.voices = cleanVoices as unknown as JsonValue
      }

      const job = await client.post<{ id: string; status: string }>('/api/jobs', {
        input_upload_id: videoUpload.id,
        isolated_tracks: isolatedTracks,
        overrides,
      })
      setMessage(`Job ${job.id} ${job.status}`)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setIsStarting(false)
    }
  }

  // All known speaker keys for voice tabs
  const allSpeakers = Array.from(
    new Set(['*', ...Object.keys(speakerToFileNameMap), ...Object.keys(voiceProfiles)])
  )
  const currentProfile = voiceProfiles[activeSpeakerTab] ?? voiceProfiles['*'] ?? DEFAULT_PROFILE
  const ttsProviders = options.tts_providers ?? ['gemini', 'openai', 'omnivoice', 'xtts', 'f5', 'bextts', 'higgs', 'coqui']
  const availableModels = options.tts_models?.[currentProfile.tts_system] ?? []
  const availableVoices = options.tts_voices?.[currentProfile.tts_system] ?? []
  const referenceCapability = options.tts_reference_capabilities?.[currentProfile.tts_system] ?? 'unsupported'
  const referenceModes = options.reference_modes ?? ['none', 'configured', 'segment', 'speaker']

  return (
    <section>
      <div className="setup-wizard-card">
        {/* Wizard Header */}
        <div className="wizard-header">
          <div>
            <h1>Workflow</h1>
            <p>Configure source media, isolated speaker tracks, and voice presets for dubbing.</p>
          </div>
          <button className="header-icon-btn" type="button" title="Close">
            <span className="material-symbols-outlined" aria-hidden="true">close</span>
          </button>
        </div>

        {/* Wizard Stepper Progress Bar */}
        <div className="wizard-stepper">
          <div className="step-item active">
            <div className="step-badge">1</div>
            <span>Setup</span>
          </div>
          <div className="step-divider"></div>
          <div className="step-item">
            <div className="step-badge">2</div>
            <span>Extraction</span>
          </div>
          <div className="step-divider"></div>
          <div className="step-item">
            <div className="step-badge">3</div>
            <span>Diarization</span>
          </div>
          <div className="step-divider"></div>
          <div className="step-item">
            <div className="step-badge">4</div>
            <span>Translation</span>
          </div>
        </div>

        {/* Wizard Form Body */}
        <div className="wizard-body">
          {error && <p role="alert">{error}</p>}
          {message && <p role="status">{message}</p>}

          <form onSubmit={submit}>
            {/* Project Details Section */}
            <div className="form-section">
              <div className="form-section-title">
                <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>folder</span>
                Project Details
              </div>
              <div className="field-grid">
                <label>
                  Project Name
                  <input
                    type="text"
                    placeholder="e.g., Q3 Marketing Video"
                    value={projectName}
                    onChange={(event) => setProjectName(event.target.value)}
                  />
                </label>
                <label>
                  Workspace
                  <select defaultValue="Personal Workspace">
                    <option value="Personal Workspace">Personal Workspace</option>
                    <option value="Team Alpha">Team Alpha</option>
                  </select>
                </label>
              </div>
            </div>

            {/* Media & Tracks Section */}
            <div className="form-section">
              <div className="form-section-title">
                <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>movie</span>
                Source Media & Isolated Tracks
              </div>

              <div className="stitch-dropzone">
                <div className="dropzone-icon">
                  <span className="material-symbols-outlined" style={{ fontSize: '28px' }}>upload_file</span>
                </div>
                <div className="dropzone-title">Drag and drop media files here</div>
                <div className="dropzone-desc">Supports MP4, MOV, WAV, MP3 up to 5GB</div>
                <div className="field-grid" style={{ width: '100%' }}>
                  <label className="field-file">
                    Video
                    <input type="file" accept="video/*" onChange={(e) => setVideo(e.target.files?.[0])} />
                  </label>
                  <label className="field-file">
                    Isolated audio track
                    <input
                      type="file"
                      accept="audio/*"
                      multiple
                      onChange={(e) => handleAddTracks(Array.from(e.target.files ?? []))}
                    />
                  </label>
                </div>
              </div>

              {/* Visual Speaker-Track Mapping Manager */}
              {tracks.length > 0 && (
                <div className="speaker-tracks-manager">
                  <div className="speaker-tracks-header">
                    <span className="speaker-tracks-title">
                      <span className="material-symbols-outlined" style={{ fontSize: '18px', color: 'var(--primary)' }}>graphic_eq</span>
                      Isolated Audio Tracks & Voice Assignment ({tracks.length})
                    </span>
                  </div>

                  <div className="speaker-tracks-list">
                    {tracks.map((file) => {
                      const speakerLabel = fileNameToSpeakerMap[file.name] || 'SPEAKER_00'
                      const speakerVoice = voiceProfiles[speakerLabel] ?? voiceProfiles['*'] ?? DEFAULT_PROFILE
                      return (
                        <div key={file.name} className="speaker-track-card">
                          <div className="speaker-track-info">
                            <span className="material-symbols-outlined speaker-track-icon">audio_file</span>
                            <div className="speaker-track-meta">
                              <span className="speaker-track-filename">{file.name}</span>
                              <span className="speaker-track-size">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                            </div>
                          </div>

                          <div className="speaker-track-assignment">
                            <label className="speaker-label-input-wrap">
                              <span className="speaker-label-tag">Speaker</span>
                              <input
                                type="text"
                                className="speaker-label-input"
                                value={speakerLabel}
                                placeholder="SPEAKER_00"
                                onChange={(e) => handleTrackSpeakerChange(file.name, e.target.value)}
                              />
                            </label>

                            <span className="speaker-arrow">➔</span>

                            {/* Direct Voice Preset Selector on Track Card */}
                            <label className="speaker-voice-select-wrap">
                              <span className="speaker-voice-tag">Dubbing Voice</span>
                              <select
                                className="speaker-voice-select"
                                value={speakerVoice.voice_name || speakerVoice.tts_system}
                                onChange={(e) => {
                                  const val = e.target.value
                                  if (val.startsWith('preset:')) {
                                    const idx = Number(val.replace('preset:', ''))
                                    handleApplyPreset(speakerLabel, idx)
                                  } else if (val.startsWith('saved:')) {
                                    const profKey = val.replace('saved:', '')
                                    handleApplySavedProfile(speakerLabel, profKey)
                                  }
                                }}
                              >
                                <optgroup label="Ready Presets">
                                  {READY_PRESETS.map((preset, idx) => (
                                    <option key={idx} value={`preset:${idx}`}>{preset.label}</option>
                                  ))}
                                </optgroup>
                                {Object.keys(savedProfiles).length > 0 && (
                                  <optgroup label="Saved Profiles">
                                    {Object.keys(savedProfiles).map((id) => (
                                      <option key={id} value={`saved:${id}`}>{id} ({savedProfiles[id]?.tts_system})</option>
                                    ))}
                                  </optgroup>
                                )}
                              </select>
                            </label>
                          </div>

                          <button
                            type="button"
                            className="speaker-track-remove-btn"
                            title="Remove track"
                            onClick={() => handleRemoveTrack(file.name)}
                          >
                            <span className="material-symbols-outlined">delete</span>
                          </button>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* Speaker-Label Mapping Textarea (Clean / Hidden for pure visual workflow, accessible for tests) */}
              <div style={{ display: 'none' }}>
                <label className="field-code">
                  Speaker-label mapping
                  <textarea
                    aria-label="Speaker-label mapping"
                    value={mapping}
                    onChange={(e) => {
                      setMapping(e.target.value)
                      try {
                        const parsed = JSON.parse(e.target.value)
                        if (isStringRecord(parsed)) {
                          setVoiceProfiles((current) => {
                            const updated = { ...current }
                            for (const spk of Object.keys(parsed)) {
                              if (!updated[spk]) updated[spk] = { ...(updated['*'] ?? DEFAULT_PROFILE) }
                            }
                            return updated
                          })
                        }
                      } catch {
                        // Keep typing
                      }
                    }}
                  />
                </label>
              </div>
            </div>

            {/* Voice Selection & Profiles Section */}
            <div className="form-section voice-settings-section">
              <div className="form-section-header-row">
                <div className="form-section-title">
                  <span className="material-symbols-outlined" style={{ fontSize: '18px', color: 'var(--primary)' }}>record_voice_over</span>
                  Dubbing Voice Selection
                </div>
                <label className="voice-toggle-label">
                  <input
                    type="checkbox"
                    checked={enableVoiceOverrides}
                    onChange={(e) => setEnableVoiceOverrides(e.target.checked)}
                  />
                  <span>Configure Dubbing Voices</span>
                </label>
              </div>

              {enableVoiceOverrides && (
                <div className="voice-editor-wrap">
                  {/* Speaker Selector Tabs */}
                  <div className="voice-speaker-tabs-bar">
                    <div className="voice-speaker-tabs" role="tablist" aria-label="Speaker voice tabs">
                      {allSpeakers.map((spk) => {
                        const isDefault = spk === '*'
                        const isActive = activeSpeakerTab === spk
                        return (
                          <button
                            key={spk}
                            type="button"
                            role="tab"
                            aria-selected={isActive}
                            className={`voice-speaker-tab${isActive ? ' active' : ''}${isDefault ? ' default-tab' : ''}`}
                            onClick={() => setActiveSpeakerTab(spk)}
                          >
                            <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>
                              {isDefault ? 'star' : 'person'}
                            </span>
                            <span>{isDefault ? 'Default Voice (*)' : spk}</span>
                            {!isDefault && (
                              <span
                                className="tab-close-icon"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  handleRemoveCustomSpeaker(spk)
                                }}
                                title="Remove voice profile"
                              >
                                ×
                              </span>
                            )}
                          </button>
                        )
                      })}
                    </div>

                    {/* Add Custom Speaker Input */}
                    <div className="add-speaker-box">
                      <input
                        type="text"
                        placeholder="Add speaker (e.g. SPEAKER_01)"
                        value={customSpeakerInput}
                        onChange={(e) => setCustomSpeakerInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            e.preventDefault()
                            handleAddCustomSpeaker()
                          }
                        }}
                      />
                      <button type="button" className="btn-add-speaker" onClick={handleAddCustomSpeaker}>
                        <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>add</span>
                        Add
                      </button>
                    </div>
                  </div>

                  {/* Active Profile Configuration Box */}
                  <div className="voice-profile-box">
                    <div className="voice-profile-box-header">
                      <div className="voice-profile-title">
                        <strong>
                          {activeSpeakerTab === '*' ? 'Default Voice (*)' : `Voice for ${activeSpeakerTab}`}
                        </strong>
                        <span className="voice-profile-subtitle">
                          {activeSpeakerTab === '*'
                            ? 'Baseline voice used for all speakers in this dubbing project'
                            : `Assigned voice preset/profile for ${activeSpeakerTab}`}
                        </span>
                      </div>

                      {/* Current Voice Badge Summary */}
                      <div className="voice-current-summary-badge">
                        <span className="badge-provider">{currentProfile.tts_system}</span>
                        {currentProfile.voice_name && <span className="badge-voice">{currentProfile.voice_name}</span>}
                        {currentProfile.model && <span className="badge-model">{currentProfile.model}</span>}
                      </div>
                    </div>

                    {/* 1-Click Quick Preset Selection */}
                    <div className="voice-quick-presets-row">
                      <label style={{ flex: 1 }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                          <span className="material-symbols-outlined" style={{ fontSize: '16px', color: 'var(--secondary)' }}>auto_awesome</span>
                          Choose from Ready Presets
                        </span>
                        <select
                          defaultValue=""
                          onChange={(e) => {
                            if (e.target.value !== '') {
                              handleApplyPreset(activeSpeakerTab, Number(e.target.value))
                              e.target.value = ''
                            }
                          }}
                        >
                          <option value="" disabled>Select a ready-made voice preset…</option>
                          {READY_PRESETS.map((preset, idx) => (
                            <option key={idx} value={idx}>{preset.label}</option>
                          ))}
                        </select>
                      </label>

                      {/* Saved Profiles from /api/voice-profiles */}
                      {Object.keys(savedProfiles).length > 0 && (
                        <label style={{ flex: 1 }}>
                          <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                            <span className="material-symbols-outlined" style={{ fontSize: '16px', color: 'var(--primary)' }}>bookmark</span>
                            Or Choose from Saved Profiles
                          </span>
                          <select
                            defaultValue=""
                            onChange={(e) => {
                              if (e.target.value) {
                                handleApplySavedProfile(activeSpeakerTab, e.target.value)
                                e.target.value = ''
                              }
                            }}
                          >
                            <option value="" disabled>Select from saved voice profiles…</option>
                            {Object.keys(savedProfiles).map((id) => (
                              <option key={id} value={id}>{id} ({savedProfiles[id]?.tts_system})</option>
                            ))}
                          </select>
                        </label>
                      )}
                    </div>

                    {/* Expandable Advanced / Detailed Configuration */}
                    <div className="voice-advanced-toggle-bar">
                      <button
                        type="button"
                        className="btn-text-action"
                        onClick={() => setShowAdvancedVoice((v) => !v)}
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>
                          {showAdvancedVoice ? 'expand_less' : 'tune'}
                        </span>
                        {showAdvancedVoice ? 'Hide detailed parameters' : 'Customize detailed parameters (TTS Provider, Model, Style Prompt)'}
                      </button>
                      <span className="voice-tab-hint">
                        💡 Manage and create custom profiles in the <strong>Voice Profiles</strong> tab
                      </span>
                    </div>

                    {showAdvancedVoice && (
                      <div className="voice-advanced-fields">
                        <div className="field-grid">
                          {/* TTS Provider */}
                          <label>
                            TTS Provider
                            <select
                              value={currentProfile.tts_system}
                              onChange={(e) => {
                                const newSystem = e.target.value
                                const defaultModel = options.tts_models?.[newSystem]?.[0] || ''
                                const defaultVoice = options.tts_voices?.[newSystem]?.[0] || ''
                                handleVoiceProfileChange(activeSpeakerTab, {
                                  tts_system: newSystem,
                                  model: defaultModel,
                                  voice_name: defaultVoice,
                                })
                              }}
                            >
                              {ttsProviders.map((prov) => (
                                <option key={prov} value={prov}>{prov}</option>
                              ))}
                            </select>
                          </label>

                          {/* Model */}
                          <label>
                            Model
                            {availableModels.length > 0 ? (
                              <select
                                value={currentProfile.model ?? ''}
                                onChange={(e) => handleVoiceProfileChange(activeSpeakerTab, { model: e.target.value })}
                              >
                                <option value="">Default / Auto</option>
                                {availableModels.map((m) => (
                                  <option key={m} value={m}>{m}</option>
                                ))}
                              </select>
                            ) : (
                              <input
                                type="text"
                                placeholder="Optional model name"
                                value={currentProfile.model ?? ''}
                                onChange={(e) => handleVoiceProfileChange(activeSpeakerTab, { model: e.target.value })}
                              />
                            )}
                          </label>

                          {/* Voice Name / Preset */}
                          {availableVoices.length > 0 && (
                            <label>
                              Voice Preset
                              <select
                                value={currentProfile.voice_name ?? ''}
                                onChange={(e) => handleVoiceProfileChange(activeSpeakerTab, { voice_name: e.target.value })}
                              >
                                <option value="">Default Voice</option>
                                {availableVoices.map((v) => (
                                  <option key={v} value={v}>{v}</option>
                                ))}
                              </select>
                            </label>
                          )}

                          {/* Reference Mode (for voice cloning providers) */}
                          {referenceCapability !== 'unsupported' && (
                            <label>
                              Reference Mode
                              <select
                                value={currentProfile.reference_mode ?? 'none'}
                                onChange={(e) => handleVoiceProfileChange(activeSpeakerTab, { reference_mode: e.target.value })}
                              >
                                {referenceModes.map((mode) => (
                                  <option key={mode} value={mode}>{mode}</option>
                                ))}
                              </select>
                            </label>
                          )}

                          {/* Reference Audio from Reference Library */}
                          {referenceCapability !== 'unsupported' && (
                            <label>
                              Reference Audio
                              {referenceEntries.length > 0 ? (
                                <select
                                  value={currentProfile.reference_audio ?? ''}
                                  onChange={(e) => handleVoiceProfileChange(activeSpeakerTab, { reference_audio: e.target.value })}
                                >
                                  <option value="">None / Auto from track</option>
                                  {referenceEntries.map((entry) => (
                                    <option key={entry.speaker_id} value={entry.audio.url}>
                                      {entry.speaker_id} ({entry.audio.name})
                                    </option>
                                  ))}
                                </select>
                              ) : (
                                <input
                                  type="text"
                                  placeholder="Path to reference wav/mp3"
                                  value={currentProfile.reference_audio ?? ''}
                                  onChange={(e) => handleVoiceProfileChange(activeSpeakerTab, { reference_audio: e.target.value })}
                                />
                              )}
                            </label>
                          )}
                        </div>

                        {/* Style Prompt / Voice Instructions */}
                        <div style={{ marginTop: '0.75rem' }}>
                          <label>
                            Style Prompt / Voice Instructions
                            <input
                              type="text"
                              placeholder="e.g. calm narrator, natural cadence, clear articulation"
                              value={currentProfile.style_prompt ?? ''}
                              onChange={(e) => handleVoiceProfileChange(activeSpeakerTab, { style_prompt: e.target.value })}
                            />
                          </label>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Job Parameters / Options */}
            {regularFields.length > 0 && (
              <div className="form-section">
                <div className="form-section-title">
                  <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>tune</span>
                  Language & Pipeline Parameters
                </div>
                <div className="field-grid">
                  {regularFields.map((field) => (
                    <WorkflowField
                      key={field.name}
                      field={field}
                      value={values[field.name]}
                      options={options}
                      onChange={(value) => setValues((current) => ({ ...current, [field.name]: value }))}
                    />
                  ))}
                </div>
                <div className="auto-detect-tag">
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>smart_toy</span>
                  Auto-detect source language available
                </div>
              </div>
            )}

            {/* Options & Flags */}
            {booleanFields.length > 0 && (
              <div className="form-section">
                <div className="form-section-title">
                  <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>toggle_on</span>
                  Pipeline Options & Flags
                </div>
                <div className="field-grid-toggles">
                  {booleanFields.map((field) => (
                    <WorkflowField
                      key={field.name}
                      field={field}
                      value={values[field.name]}
                      options={options}
                      onChange={(value) => setValues((current) => ({ ...current, [field.name]: value }))}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Wizard Footer Action Bar */}
            <div className="wizard-footer">
              <button className="cancel-btn" type="button">Cancel</button>
              <button type="submit" className="next-step-btn" disabled={isStarting}>
                {isStarting ? 'Starting…' : 'Start'}
                <span className="material-symbols-outlined" aria-hidden="true" style={{ fontSize: '18px' }}>arrow_forward</span>
              </button>
            </div>
          </form>
        </div>
      </div>
    </section>
  )
}

function WorkflowField({ field, value, options, onChange }: { field: SchemaFieldDefinition; value: JsonValue; options: OptionsResponse; onChange(value: JsonValue): void }) {
  const label = field.label ?? field.name
  if (field.type === 'boolean') {
    return (
      <label className="field-boolean">
        <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
        {label}
      </label>
    )
  }
  if (field.type === 'select') {
    const choices = normalizeOptions(field.options ?? options[field.options_key ?? field.name])
    return (
      <label className="field-select">
        {label}
        <select value={String(value ?? '')} onChange={(e) => onChange(e.target.value)}>
          {choices.map((choice) => (
            <option key={choice.value} value={choice.value}>{choice.label}</option>
          ))}
        </select>
      </label>
    )
  }
  return (
    <label className="field-input">
      {label}
      <input
        type={field.type === 'number' ? 'number' : 'text'}
        value={String(value ?? '')}
        onChange={(e) => onChange(field.type === 'number' ? Number(e.target.value) : e.target.value)}
      />
    </label>
  )
}

function normalizeOptions(value: JsonValue | Array<SelectOption | string> | undefined): SelectOption[] {
  if (!Array.isArray(value)) return []
  return value.map((item) => typeof item === 'string' ? { value: item, label: item } : item as unknown as SelectOption)
}

async function upload(client: ApiClient, file: File) {
  const form = new FormData()
  form.append('file', file)
  return client.upload<{ id: string }>('/api/uploads', form)
}

function isStringRecord(value: unknown): value is Record<string, string> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    && Object.entries(value).every(([speaker, fileName]) => speaker.length > 0 && typeof fileName === 'string' && fileName.length > 0)
}

function deduceSpeakerLabel(fileName: string, index: number): string {
  const base = fileName.replace(/\.[^/.]+$/, '').trim()
  const speakerMatch = base.match(/^speaker_?(\d+)$/i)
  if (speakerMatch) {
    const num = parseInt(speakerMatch[1], 10)
    return `SPEAKER_${num < 10 ? '0' + num : num}`
  }
  const trackMatch = base.match(/^track_?(\d+)$/i)
  if (trackMatch) {
    const num = parseInt(trackMatch[1], 10)
    return `SPEAKER_${num < 10 ? '0' + num : num}`
  }
  if (/^[a-zA-Z0-9_-]+$/.test(base) && base.length <= 20) {
    return base
  }
  return `SPEAKER_${index < 10 ? '0' + index : index}`
}
