import { FormEvent, useEffect, useState } from 'react'
import type { ApiClient, SelectOption } from '../api/types'

interface Profile { id: string; name: string; model: string; voice: string; reference_id?: string }
interface Reference { id: string; name: string }

export function VoicesView({ client }: { client: ApiClient }) {
  const [profiles, setProfiles] = useState<Profile[]>([]); const [references, setReferences] = useState<Reference[]>([])
  const [models, setModels] = useState<SelectOption[]>([]); const [voices, setVoices] = useState<SelectOption[]>([])
  const [profile, setProfile] = useState<Profile>({ id: '', name: '', model: '', voice: '' }); const [referenceFile, setReferenceFile] = useState<File>()
  const [speaker, setSpeaker] = useState(''); const [error, setError] = useState('')
  useEffect(() => { Promise.all([client.get<{ profiles: Profile[] }>('/api/voice-profiles'), client.get<{ references: Reference[] }>('/api/references'), client.get<{ voice_models: SelectOption[]; voices: SelectOption[] }>('/api/options')]).then(([profileData, referenceData, optionData]) => {
    setProfiles(profileData.profiles); setReferences(referenceData.references); setModels(optionData.voice_models); setVoices(optionData.voices); if (profileData.profiles[0]) setProfile(profileData.profiles[0])
  }).catch((reason) => setError(String(reason))) }, [client])
  const compatibleVoices = voices.filter((voice) => !voice.depends_on?.model || voice.depends_on.model === profile.model)

  async function save(event: FormEvent) { event.preventDefault(); const id = profile.id || 'new'; await client.put(`/api/voice-profiles/${encodeURIComponent(id)}`, profile) }
  async function upload(event: FormEvent) { event.preventDefault(); if (!referenceFile) return; const data = new FormData(); data.append('file', referenceFile); const created = await client.upload<Reference>('/api/references', data); setReferences((current) => [...current, created]) }

  return <section><h1>Voice Profiles</h1>{error && <p role="alert">{error}</p>}<form onSubmit={save}>
    <label>Profile<select value={profile.id} onChange={(e) => setProfile(profiles.find((item) => item.id === e.target.value) ?? profile)}>{profiles.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
    <label>Name<input value={profile.name} onChange={(e) => setProfile({ ...profile, name: e.target.value })} /></label>
    <label>Model<select value={profile.model} onChange={(e) => setProfile({ ...profile, model: e.target.value, voice: '' })}>{models.map(option)}</select></label>
    <label>Voice<select value={profile.voice} onChange={(e) => setProfile({ ...profile, voice: e.target.value })}>{compatibleVoices.map(option)}</select></label>
    <label>Reference<select value={profile.reference_id ?? ''} onChange={(e) => setProfile({ ...profile, reference_id: e.target.value })}><option value="">None</option>{references.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
    <div className="actions"><button type="submit">Save profile</button><button type="button" onClick={() => client.delete(`/api/voice-profiles/${encodeURIComponent(profile.id)}`)}>Delete profile</button></div>
  </form>
  <h2>References</h2><form onSubmit={upload}><label>Reference audio<input type="file" accept="audio/*" onChange={(e) => setReferenceFile(e.target.files?.[0])} /></label><button type="submit">Upload reference</button></form>
  <ul>{references.map((item) => <li key={item.id}>{item.name} <button type="button" onClick={() => client.delete(`/api/references/${encodeURIComponent(item.id)}`)}>Delete {item.name}</button></li>)}</ul>
  <h2>Assignment</h2><label>Speaker label<input value={speaker} onChange={(e) => setSpeaker(e.target.value)} /></label><button type="button" onClick={() => client.put(`/api/voice-assignments/${encodeURIComponent(speaker)}`, { profile_id: profile.id })}>Assign profile</button>
  </section>
}

function option(item: SelectOption) { return <option key={item.value} value={item.value}>{item.label}</option> }
