import { FormEvent, useEffect, useRef, useState } from 'react'
import type { ApiClient } from '../api/types'

import type { VoiceProfile, ProfilesSnapshot, ReferenceEntry, ReferencesSnapshot, ExtendedOptions as VoiceOptions } from '../types/models'


const emptyProfile: VoiceProfile = { tts_system: '' }

function findMatchingReference(savedRef: string | undefined, refs: ReferenceEntry[]): string {
  if (!savedRef || refs.length === 0) return ''
  const normSaved = savedRef.replace(/_/g, ' ').trim().toLowerCase()
  const exact = refs.find(e => e.speaker_id === savedRef)
  if (exact) return exact.speaker_id
  const byNorm = refs.find(e => e.speaker_id.replace(/_/g, ' ').trim().toLowerCase() === normSaved)
  if (byNorm) return byNorm.speaker_id
  const byUrl = refs.find(e => e.audio.url === savedRef)
  if (byUrl) return byUrl.speaker_id
  const byId = refs.find(e => e.audio.id && savedRef.includes(e.audio.id))
  if (byId) return byId.speaker_id
  const byName = refs.find(e => e.audio.name && (savedRef === e.audio.name || savedRef.endsWith('/' + e.audio.name) || savedRef.endsWith('\\' + e.audio.name)))
  if (byName) return byName.speaker_id
  return ''
}

export function VoicesView({ client }: { client: ApiClient }) {
  const [profiles, setProfiles] = useState<Record<string, VoiceProfile>>({})
  const [profilesRevision, setProfilesRevision] = useState('')
  const [references, setReferences] = useState<ReferenceEntry[]>([])
  const [referencesRevision, setReferencesRevision] = useState('')
  const [options, setOptions] = useState<VoiceOptions>({ tts_providers: [], tts_models: {}, tts_voices: {} })
  const [profileId, setProfileId] = useState('')
  const [profile, setProfile] = useState<VoiceProfile>(emptyProfile)
  const [selectedReference, setSelectedReference] = useState('')
  const [referenceFile, setReferenceFile] = useState<File>()
  const [referenceText, setReferenceText] = useState('')
  const [speaker, setSpeaker] = useState('')
  const [error, setError] = useState('')
  const [showAddForm, setShowAddForm] = useState(false)
  const [editingRef, setEditingRef] = useState<string | null>(null)
  const [editText, setEditText] = useState('')
  const [playingId, setPlayingId] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    Promise.all([
      client.get<ProfilesSnapshot>('/api/voice-profiles'),
      client.get<ReferencesSnapshot>('/api/reference-library'),
      client.get<VoiceOptions>('/api/options'),
    ]).then(([profileData, referenceData, optionData]) => {
      setProfiles(profileData.profiles ?? {})
      setProfilesRevision(profileData.revision)
      const entries = Array.isArray(referenceData.entries) ? referenceData.entries : []
      setReferences(entries)
      setReferencesRevision(referenceData.revision)
      setOptions(optionData)
      const first = Object.entries(profileData.profiles ?? {})[0]
      if (first) {
        setProfileId(first[0]); setProfile(first[1])
        const savedRef = first[1].reference_audio
        const matched = findMatchingReference(savedRef, entries)
        setSelectedReference(matched || (first[1].reference_mode === 'configured' ? entries[0]?.speaker_id ?? '' : ''))
      } else {
        setSelectedReference('')
      }
    }).catch(showError)
  }, [client])

  function showError(reason: unknown) { setError(reason instanceof Error ? reason.message : String(reason)) }
  function selectProfile(id: string) {
    const p = profiles[id] ?? emptyProfile
    setProfileId(id); setProfile(p)
    // sync selectedReference with the profile's saved reference_audio
    const savedRef = p.reference_audio
    const matched = findMatchingReference(savedRef, references)
    if (matched) {
      setSelectedReference(matched)
    } else if (p.reference_mode === 'configured' && references.length > 0) {
      setSelectedReference(references[0].speaker_id)
    } else {
      setSelectedReference('')
    }
  }
  function startNewProfile() {
    setError('')
    setSelectedReference('')
    const count = Object.keys(profiles).length
    let nextId = `SPEAKER_${String(count).padStart(2, '0')}`
    let i = count
    while (profiles[nextId]) {
      i++
      nextId = `SPEAKER_${String(i).padStart(2, '0')}`
    }
    setProfileId(nextId)
    const defaultProvider = options.tts_providers?.[0] ?? ''
    const models = options.tts_models?.[defaultProvider] ?? []
    const voices = options.tts_voices?.[defaultProvider] ?? []
    const cap = options.tts_reference_capabilities?.[defaultProvider] ?? 'unsupported'
    setProfile({
      tts_system: defaultProvider,
      model: models[0] || undefined,
      voice_name: voices[0] || undefined,
      reference_mode: cap === 'required' ? 'speaker' : cap === 'optional' ? 'none' : undefined,
    })
  }

  function handleProviderChange(provider: string) {
    const cap = options.tts_reference_capabilities?.[provider] ?? 'unsupported'
    const models = options.tts_models?.[provider] ?? []
    const voices = options.tts_voices?.[provider] ?? []
    setProfile({
      tts_system: provider,
      model: models[0] || undefined,
      voice_name: voices[0] || undefined,
      reference_mode: cap === 'required' ? (profile.reference_mode && profile.reference_mode !== 'none' ? profile.reference_mode : (selectedReference ? 'configured' : 'speaker')) : cap === 'optional' ? (profile.reference_mode ?? (selectedReference ? 'configured' : 'none')) : undefined,
      reference_audio: cap !== 'unsupported' ? (selectedReference || profile.reference_audio) : undefined,
    })
  }

  function applyProfiles(snapshot: ProfilesSnapshot, selected = profileId) {
    setProfiles(snapshot.profiles); setProfilesRevision(snapshot.revision)
    if (snapshot.profiles[selected]) {
      setProfileId(selected)
      setProfile(snapshot.profiles[selected])
    } else {
      const entries = Object.entries(snapshot.profiles ?? {})
      if (entries.length > 0) {
        setProfileId(entries[0][0])
        setProfile(entries[0][1])
      } else {
        setProfileId('')
        setProfile(emptyProfile)
      }
    }
  }
  function applyReferences(snapshot: ReferencesSnapshot) {
    setReferences(snapshot.entries); setReferencesRevision(snapshot.revision)
    if (!snapshot.entries.some((item) => item.speaker_id === selectedReference)) setSelectedReference(snapshot.entries[0]?.speaker_id ?? '')
  }

  async function save(event: FormEvent) {
    event.preventDefault(); setError(''); setSaving(true)
    const trimmedId = profileId.trim()
    if (!trimmedId) {
      setError('Profile name is required')
      setSaving(false)
      return
    }
    const cap = options.tts_reference_capabilities?.[profile.tts_system] ?? 'unsupported'
    const payloadProfile: VoiceProfile = {
      tts_system: profile.tts_system,
      model: profile.model || undefined,
      voice_name: profile.voice_name || undefined,
    }
    if (cap !== 'unsupported') {
      const mode = selectedReference
        ? 'configured'
        : (profile.reference_mode || (cap === 'required' ? 'speaker' : 'none'))
      payloadProfile.reference_mode = mode
      if (mode === 'configured') {
        const refAudio = selectedReference || profile.reference_audio
        if (refAudio) payloadProfile.reference_audio = refAudio
      }
    }
    try {
      const snapshot = await client.put<ProfilesSnapshot>(`/api/voice-profiles/${encodeURIComponent(trimmedId)}`, { revision: profilesRevision, profile: payloadProfile })
      applyProfiles(snapshot, trimmedId)
    } catch (reason) { showError(reason) }
    setSaving(false)
  }
  async function removeProfile() {
    if (!profileId || !profiles[profileId]) return
    try { applyProfiles(await client.delete<ProfilesSnapshot>(`/api/voice-profiles/${encodeURIComponent(profileId)}`, { revision: profilesRevision })) }
    catch (reason) { showError(reason) }
  }
  async function uploadReference(event: FormEvent) {
    event.preventDefault(); if (!referenceFile || !speaker) return
    setUploading(true)
    const data = new FormData(); data.append('speaker_id', speaker); data.append('reference_text', referenceText); data.append('revision', referencesRevision); data.append('file', referenceFile)
    try {
      applyReferences(await client.upload<ReferencesSnapshot>('/api/reference-library', data))
      setShowAddForm(false); setSpeaker(''); setReferenceText(''); setReferenceFile(undefined)
    }
    catch (reason) { showError(reason) }
    setUploading(false)
  }
  async function removeReference(id: string) {
    try { applyReferences(await client.delete<ReferencesSnapshot>(`/api/reference-library/${encodeURIComponent(id)}`, { revision: referencesRevision })) }
    catch (reason) { showError(reason) }
    setDeleteConfirm(null)
  }


  function togglePlay(entry: ReferenceEntry) {
    if (playingId === entry.speaker_id) {
      audioRef.current?.pause()
      setPlayingId(null)
    } else {
      if (audioRef.current) { audioRef.current.pause(); audioRef.current.src = entry.audio.url; audioRef.current.play().catch(() => {}) }
      setPlayingId(entry.speaker_id)
    }
  }

  function startEdit(entry: ReferenceEntry) { setEditingRef(entry.speaker_id); setEditText(entry.reference_text) }
  function cancelEdit() { setEditingRef(null); setEditText('') }

  const models = options.tts_models?.[profile.tts_system] ?? []
  const voices = options.tts_voices?.[profile.tts_system] ?? []
  const referenceCapability = options.tts_reference_capabilities?.[profile.tts_system] ?? 'unsupported'
  const referenceModes = referenceCapability === 'required'
    ? ['speaker', 'segment', 'configured']
    : referenceCapability === 'optional'
    ? ['none', 'speaker', 'segment', 'configured']
    : []

  return (
    <section className="voices-view">
      <h1 className="sr-only">Voice Profiles</h1>
      <audio ref={audioRef} onEnded={() => setPlayingId(null)} style={{ display: 'none' }} />

      {error && (
        <div className="voices-error" role="alert">
          <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>error</span>
          {error}
          <button type="button" onClick={() => setError('')} className="voices-error-close">
            <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>close</span>
          </button>
        </div>
      )}

      {/* ── Voice Profile Configuration ── */}
      <div className="voices-block">
        <div className="voices-block-header">
          <div className="voices-block-title">
            <span className="material-symbols-outlined">mic</span>
            Voice Profiles
          </div>
          <button type="button" className="btn-ghost-sm" onClick={startNewProfile}>
            <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>add</span>
            New profile
          </button>
        </div>

        <form onSubmit={save}>
          <div className="voices-profile-grid">
            <label className="voices-label">
              <span className="voices-label-text">Profile</span>
              <select value={profileId in profiles ? profileId : ''} onChange={(e) => selectProfile(e.target.value)}>
                {!profiles[profileId] && <option value="">{profileId ? `+ ${profileId} (New)` : '-- New Profile --'}</option>}
                {Object.keys(profiles).map((id) => <option key={id} value={id}>{id}</option>)}
              </select>
            </label>
            <label className="voices-label">
              <span className="voices-label-text">Profile name</span>
              <input value={profileId} onChange={(e) => setProfileId(e.target.value)} placeholder="e.g. John Male, Narrator, *" />
            </label>
            <label className="voices-label">
              <span className="voices-label-text">Provider</span>
              <select value={profile.tts_system} onChange={(e) => handleProviderChange(e.target.value)}>
                {options.tts_providers?.map(opt)}
              </select>
            </label>
            <label className="voices-label">
              <span className="voices-label-text">Model</span>
              <select value={profile.model ?? ''} onChange={(e) => setProfile({ ...profile, model: e.target.value })}>
                <option value="">None</option>{models.map(opt)}
              </select>
            </label>
            <label className="voices-label">
              <span className="voices-label-text">Voice</span>
              <select value={profile.voice_name ?? ''} onChange={(e) => setProfile({ ...profile, voice_name: e.target.value })}>
                <option value="">None</option>{voices.map(opt)}
              </select>
            </label>
            {referenceCapability !== 'unsupported' && (
              <label className="voices-label">
                <span className="voices-label-text">Reference Mode</span>
                <select
                  value={profile.reference_mode ?? (selectedReference ? 'configured' : (referenceCapability === 'required' ? 'speaker' : 'none'))}
                  onChange={(e) => {
                    const mode = e.target.value
                    setProfile({ ...profile, reference_mode: mode })
                    if (mode === 'configured' && !selectedReference && references.length > 0) {
                      setSelectedReference(references[0].speaker_id)
                    } else if (mode !== 'configured') {
                      setSelectedReference('')
                    }
                  }}
                >
                  {referenceModes.map(opt)}
                </select>
              </label>
            )}
            <label className="voices-label">
              <span className="voices-label-text">Reference Audio</span>
              <select
                value={selectedReference}
                onChange={(e) => {
                  const ref = e.target.value
                  setSelectedReference(ref)
                  if (ref) {
                    setProfile((prev) => ({ ...prev, reference_mode: 'configured', reference_audio: ref }))
                  } else {
                    setProfile((prev) => ({ ...prev, reference_mode: referenceCapability === 'required' ? 'speaker' : 'none', reference_audio: undefined }))
                  }
                }}
              >
                <option value="">None</option>
                {references.map((item) => <option key={item.speaker_id} value={item.speaker_id}>{item.speaker_id}</option>)}
              </select>
            </label>
          </div>

          <div className="voices-profile-actions">
            <button type="submit" className="btn-primary" disabled={saving}>
              {saving ? <><span className="material-symbols-outlined spin" style={{ fontSize: '15px' }}>progress_activity</span> Saving…</> : <><span className="material-symbols-outlined" style={{ fontSize: '15px' }}>save</span> Save profile</>}
            </button>
            <button type="button" className="btn-ghost" onClick={removeProfile} disabled={!profiles[profileId]}>
              <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>delete</span> Delete
            </button>
          </div>
        </form>
      </div>

      {/* ── Reference Library ── */}
      <div className="voices-block">
        <div className="voices-block-header">
          <div className="voices-block-title">
            <span className="material-symbols-outlined">library_music</span>
            Reference Library
            {references.length > 0 && <span className="ref-count-badge">{references.length}</span>}
          </div>
          <button type="button" className="btn-ghost-sm" onClick={() => { setShowAddForm(!showAddForm); setError('') }}>
            <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>{showAddForm ? 'close' : 'add'}</span>
            {showAddForm ? 'Cancel' : 'Add reference'}
          </button>
        </div>

        {/* Add new reference form */}
        {showAddForm && (
          <form onSubmit={uploadReference} className="ref-add-form">
            <div className="ref-add-form-title">
              <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>upload_file</span>
              Upload New Reference
            </div>
            <div className="ref-add-grid">
              <label className="voices-label">
                <span className="voices-label-text">Reference name</span>
                <input value={speaker} onChange={(e) => setSpeaker(e.target.value)} placeholder="e.g. John Voice, Scene A" required />
              </label>
              <label className="voices-label">
                <span className="voices-label-text">Reference text <span className="voices-label-opt">(optional)</span></span>
                <input value={referenceText} onChange={(e) => setReferenceText(e.target.value)} placeholder="Transcript of the audio…" />
              </label>
              <label className="voices-label ref-file-label">
                <span className="voices-label-text">Audio file</span>
                <div className="ref-file-input-wrap">
                  <input type="file" accept="audio/*" onChange={(e) => setReferenceFile(e.target.files?.[0])} required />
                </div>
              </label>
            </div>
            <div className="voices-profile-actions">
              <button type="submit" className="btn-primary" disabled={uploading || !referenceFile || !speaker}>
                {uploading
                  ? <><span className="material-symbols-outlined spin" style={{ fontSize: '15px' }}>progress_activity</span> Uploading…</>
                  : <><span className="material-symbols-outlined" style={{ fontSize: '15px' }}>cloud_upload</span> Upload reference</>}
              </button>
              <button type="button" className="btn-ghost" onClick={() => { setShowAddForm(false); setSpeaker(''); setReferenceText(''); setReferenceFile(undefined) }}>
                Cancel
              </button>
            </div>
          </form>
        )}

        {/* Reference entries list */}
        {references.length === 0 && !showAddForm ? (
          <div className="ref-empty">
            <span className="material-symbols-outlined ref-empty-icon">library_music</span>
            <p className="ref-empty-title">No references yet</p>
            <p className="ref-empty-desc">Upload a reference audio file to get started</p>
            <button type="button" className="btn-primary" onClick={() => setShowAddForm(true)}>
              <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>add</span>
              Add first reference
            </button>
          </div>
        ) : (
          <div className="ref-list">
            {references.map((entry) => (
              <div key={entry.speaker_id} className={`ref-card${editingRef === entry.speaker_id ? ' ref-card--editing' : ''}`}>
                {/* Card header */}
                <div className="ref-card-header">
                  <div className="ref-card-meta">
                    <span className="ref-speaker-badge">{entry.speaker_id}</span>
                    <span className="ref-file-name">{entry.audio.name}</span>
                  </div>
                  <div className="ref-card-actions">
                    {/* Play/pause */}
                    <button
                      type="button"
                      className={`ref-icon-btn${playingId === entry.speaker_id ? ' ref-icon-btn--active' : ''}`}
                      onClick={() => togglePlay(entry)}
                      title={playingId === entry.speaker_id ? 'Pause' : 'Play'}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>
                        {playingId === entry.speaker_id ? 'pause_circle' : 'play_circle'}
                      </span>
                    </button>
                    {/* Download */}
                    <a
                      href={entry.audio.url}
                      download={entry.audio.name}
                      className="ref-icon-btn"
                      title="Download"
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>download</span>
                    </a>
                    {/* Edit text */}
                    <button
                      type="button"
                      className={`ref-icon-btn${editingRef === entry.speaker_id ? ' ref-icon-btn--active' : ''}`}
                      onClick={() => editingRef === entry.speaker_id ? cancelEdit() : startEdit(entry)}
                      title="Edit reference text"
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>
                        {editingRef === entry.speaker_id ? 'edit_off' : 'edit'}
                      </span>
                    </button>
                    {/* Delete */}
                    {deleteConfirm === entry.speaker_id ? (
                      <div className="ref-delete-confirm">
                        <span className="ref-delete-confirm-text">Delete?</span>
                        <button type="button" className="ref-icon-btn ref-icon-btn--danger" onClick={() => removeReference(entry.speaker_id)} title="Confirm delete">
                          <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>check</span>
                        </button>
                        <button type="button" className="ref-icon-btn" onClick={() => setDeleteConfirm(null)} title="Cancel">
                          <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>close</span>
                        </button>
                      </div>
                    ) : (
                      <button type="button" className="ref-icon-btn ref-icon-btn--danger-hover" onClick={() => setDeleteConfirm(entry.speaker_id)} title="Delete">
                        <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>delete</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* Reference text — view or edit */}
                {editingRef === entry.speaker_id ? (
                  <div className="ref-edit-area">
                    <textarea
                      className="ref-text-input"
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      placeholder="Enter reference transcript…"
                      rows={3}
                      autoFocus
                    />
                    <div className="ref-edit-actions">
                      <button
                        type="button"
                        className="btn-primary"
                        style={{ fontSize: '0.8125rem', padding: '0.375rem 0.875rem' }}
                        onClick={async () => {
                          try {
                            const data = new FormData()
                            data.append('speaker_id', entry.speaker_id)
                            data.append('reference_text', editText)
                            data.append('revision', referencesRevision)
                            applyReferences(await client.upload<ReferencesSnapshot>('/api/reference-library', data))
                            cancelEdit()
                          } catch (reason) { showError(reason) }
                        }}
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>check</span> Save text
                      </button>
                      <button type="button" className="btn-ghost" style={{ fontSize: '0.8125rem', padding: '0.375rem 0.875rem' }} onClick={cancelEdit}>Cancel</button>
                    </div>
                  </div>
                ) : entry.reference_text ? (
                  <p className="ref-text-display">{entry.reference_text}</p>
                ) : (
                  <p className="ref-text-empty">
                    <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>edit</span>
                    No transcript — click edit to add one
                  </p>
                )}

                {/* Playing waveform indicator */}
                {playingId === entry.speaker_id && (
                  <div className="ref-playing-bar">
                    <span className="material-symbols-outlined" style={{ fontSize: '13px', color: 'var(--primary)' }}>graphic_eq</span>
                    <span className="ref-playing-label">Playing…</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}

function opt(value: string) { return <option key={value} value={value}>{value}</option> }
