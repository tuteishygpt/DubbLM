import { FormEvent, useEffect, useState } from 'react'
import type { ApiClient } from '../api/types'

interface VoiceProfile { tts_system: string; model?: string; voice_name?: string }
interface ProfilesSnapshot { revision: string; profiles: Record<string, VoiceProfile> }
interface ReferenceEntry { speaker_id: string; reference_text: string; audio: { id: string; name: string; url: string } }
interface ReferencesSnapshot { revision: string; entries: ReferenceEntry[] }
interface VoiceOptions { tts_providers: string[]; tts_models: Record<string, string[]>; tts_voices: Record<string, string[]> }

const emptyProfile: VoiceProfile = { tts_system: '' }

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

  useEffect(() => {
    Promise.all([
      client.get<ProfilesSnapshot>('/api/voice-profiles'),
      client.get<ReferencesSnapshot>('/api/reference-library'),
      client.get<VoiceOptions>('/api/options'),
    ]).then(([profileData, referenceData, optionData]) => {
      setProfiles(profileData.profiles ?? {})
      setProfilesRevision(profileData.revision)
      setReferences(Array.isArray(referenceData.entries) ? referenceData.entries : [])
      setReferencesRevision(referenceData.revision)
      setOptions(optionData)
      const first = Object.entries(profileData.profiles ?? {})[0]
      if (first) { setProfileId(first[0]); setProfile(first[1]) }
      const firstReference = referenceData.entries?.[0]
      if (firstReference) setSelectedReference(firstReference.speaker_id)
    }).catch(showError)
  }, [client])

  function showError(reason: unknown) { setError(reason instanceof Error ? reason.message : String(reason)) }
  function selectProfile(id: string) { setProfileId(id); setProfile(profiles[id] ?? emptyProfile) }
  function applyProfiles(snapshot: ProfilesSnapshot, selected = profileId) {
    setProfiles(snapshot.profiles); setProfilesRevision(snapshot.revision)
    if (snapshot.profiles[selected]) setProfile(snapshot.profiles[selected])
  }
  function applyReferences(snapshot: ReferencesSnapshot) {
    setReferences(snapshot.entries); setReferencesRevision(snapshot.revision)
    if (!snapshot.entries.some((item) => item.speaker_id === selectedReference)) setSelectedReference(snapshot.entries[0]?.speaker_id ?? '')
  }

  async function save(event: FormEvent) {
    event.preventDefault(); setError('')
    try {
      const snapshot = await client.put<ProfilesSnapshot>(`/api/voice-profiles/${encodeURIComponent(profileId)}`, { revision: profilesRevision, profile })
      applyProfiles(snapshot)
    } catch (reason) { showError(reason) }
  }
  async function removeProfile() {
    try { applyProfiles(await client.delete<ProfilesSnapshot>(`/api/voice-profiles/${encodeURIComponent(profileId)}`, { revision: profilesRevision })) }
    catch (reason) { showError(reason) }
  }
  async function uploadReference(event: FormEvent) {
    event.preventDefault(); if (!referenceFile || !speaker) return
    const data = new FormData(); data.append('speaker_id', speaker); data.append('reference_text', referenceText); data.append('revision', referencesRevision); data.append('file', referenceFile)
    try { applyReferences(await client.upload<ReferencesSnapshot>('/api/reference-library', data)) }
    catch (reason) { showError(reason) }
  }
  async function removeReference(id: string) {
    try { applyReferences(await client.delete<ReferencesSnapshot>(`/api/reference-library/${encodeURIComponent(id)}`, { revision: referencesRevision })) }
    catch (reason) { showError(reason) }
  }
  async function assignReference() {
    if (!selectedReference || !speaker) return
    try {
      const result = await client.put<{ settings_revision: string }>(`/api/reference-library/${encodeURIComponent(selectedReference)}`, { profile_speaker_id: speaker, settings_revision: profilesRevision })
      setProfilesRevision(result.settings_revision)
    } catch (reason) { showError(reason) }
  }

  const models = options.tts_models?.[profile.tts_system] ?? []
  const voices = options.tts_voices?.[profile.tts_system] ?? []
  return <section><h1>Voice Profiles</h1>{error && <p role="alert">{error}</p>}<form onSubmit={save}>
    <label>Profile<select value={profileId} onChange={(e) => selectProfile(e.target.value)}>{Object.keys(profiles).map((id) => <option key={id} value={id}>{id}</option>)}</select></label>
    <label>Name<input value={profileId} onChange={(e) => setProfileId(e.target.value)} /></label>
    <label>Provider<select value={profile.tts_system} onChange={(e) => setProfile({ tts_system: e.target.value })}>{options.tts_providers?.map(option)}</select></label>
    <label>Model<select value={profile.model ?? ''} onChange={(e) => setProfile({ ...profile, model: e.target.value })}><option value="">None</option>{models.map(option)}</select></label>
    <label>Voice<select value={profile.voice_name ?? ''} onChange={(e) => setProfile({ ...profile, voice_name: e.target.value })}><option value="">None</option>{voices.map(option)}</select></label>
    <label>Reference<select value={selectedReference} onChange={(e) => setSelectedReference(e.target.value)}><option value="">None</option>{references.map((item) => <option key={item.speaker_id} value={item.speaker_id}>{item.audio.name}</option>)}</select></label>
    <div className="actions"><button type="submit">Save profile</button><button type="button" onClick={removeProfile}>Delete profile</button></div>
  </form>
  <h2>References</h2><label>Speaker label<input value={speaker} onChange={(e) => setSpeaker(e.target.value)} /></label>
  <form onSubmit={uploadReference}><label>Reference audio<input type="file" accept="audio/*" onChange={(e) => setReferenceFile(e.target.files?.[0])} /></label><label>Reference text<input value={referenceText} onChange={(e) => setReferenceText(e.target.value)} /></label><button type="submit">Upload reference</button></form>
  <ul>{references.map((item) => <li key={item.speaker_id}><a href={item.audio.url}>{item.audio.name}</a> <button type="button" onClick={() => removeReference(item.speaker_id)}>Delete {item.audio.name}</button></li>)}</ul>
  <h2>Assignment</h2><button type="button" onClick={assignReference}>Assign profile</button>
  </section>
}

function option(value: string) { return <option key={value} value={value}>{value}</option> }
