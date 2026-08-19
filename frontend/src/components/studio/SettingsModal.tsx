import type { VoiceProfile, Job } from '../../types/models'
import type { ConfigResponse, JsonValue, OptionsResponse, SchemaFieldDefinition } from '../../api/types'
import { SchemaField } from '../SchemaField'
import type { ApiClient } from '../../api/types'

function normalizeOptions(value: unknown): any[] {
  return Array.isArray(value) ? value.map((item) => typeof item === 'string' ? { value: item, label: item } : item) : []
}

function getFieldSection(fieldName: string): string {
  if (fieldName.includes('llm') || fieldName.includes('translator') || fieldName.includes('refinement') || fieldName === 'translation_prompt_prefix' || fieldName === 'glossary') return 'LLM & Translation';
  if (fieldName.includes('transcription')) return 'Transcription & Diarization';
  if (fieldName.includes('tts') || fieldName.includes('voice') || fieldName === 'segment_reference_min_duration') return 'Voice Generation (TTS)';
  if (fieldName.includes('emotion')) return 'Emotions & Context';
  if (fieldName.includes('timing') || fieldName.includes('volume') || fieldName.includes('pause') || fieldName.includes('semantic_split') || fieldName === 'keyframe_buffer' || fieldName === 'use_two_pass_encoding' || fieldName === 'keep_original_audio_ranges') return 'Timing & Mixing';
  if (fieldName.includes('debug') || fieldName.includes('watermark')) return 'Debug & Watermarks';
  return 'General Execution';
}

import { useStudio } from '../../contexts/StudioContext'

export function SettingsModal() {
  const {
    showSettingsPanel: show,
    onCloseSettingsPanel: onClose,
    activeSettingsTab: activeTab,
    setActiveSettingsTab: onTabChange,
    displaySpeakers,
    voiceProfiles,
    speakerMap,
    setSpeakerMap: onSpeakerMapChange,
    speakerMapStatus,
    globalConfig,
    globalOptions,
    globalSettingsValues,
    setGlobalSettingsValues: onGlobalSettingsValuesChange,
    globalSettingsSaving,
    globalSettingsStatus,
    handleSaveSpeakerMap: onSaveSpeakerMap,
    client,
    projects,
    jobs,
    selectedJobId,
    selectedProjectName,
    setGlobalSettingsStatus,
    setGlobalConfig,
    setGlobalSettingsSaving,
    speakerMapSaving,
  } = useStudio()

  if (!show) return null

  // Define the save handler locally in SettingsModal since it needs specific fields
  const onSaveGlobalSettings = async () => {
    if (!client || !globalConfig) return
    setGlobalSettingsSaving(true)
    try {
      const normalized = { ...globalSettingsValues }
      for (const field of globalConfig.schema.fields) {
        if (['list', 'object', 'structured'].includes(field.type) && typeof normalized[field.name] === 'string') {
          try { normalized[field.name] = JSON.parse(normalized[field.name] as string) as JsonValue } catch (e) {}
        }
      }
      const updated = await client.put<ConfigResponse>('/api/config', { revision: globalConfig.revision, values: normalized })
      
      const projectName = selectedProjectName || projects.find((p: any) => p.job_id === selectedJobId)?.name || jobs.find((j: any) => j.id === selectedJobId)?.project_name
      if (projectName) {
        try {
          await client.put(`/api/projects/${encodeURIComponent(projectName)}/config`, { values: normalized })
        } catch (err) {
          console.warn('Failed to sync settings to project metadata:', err)
        }
      }

      setGlobalConfig(updated)
      onGlobalSettingsValuesChange(updated.values)
      setGlobalSettingsStatus(projectName ? 'Settings saved to global and active project.' : 'Settings applied & saved.')
    } catch (e) {
      setGlobalSettingsStatus(e instanceof Error ? e.message : 'Failed to save.')
    } finally {
      setGlobalSettingsSaving(false)
      setTimeout(() => setGlobalSettingsStatus(''), 3500)
    }
  }

  return (
    <>
      <div className="settings-modal-overlay" onClick={onClose} aria-hidden="true" />
      <div className="settings-modal-panel" role="dialog" aria-label="Project Settings" aria-modal="true">
        <div className="settings-modal-header">
          <div className="settings-modal-title">
            <span className="material-symbols-outlined" style={{ color: 'var(--primary)' }}>tune</span>
            <span>Project & Pipeline Settings</span>
          </div>
          <button type="button" className="settings-modal-close" onClick={onClose} aria-label="Close settings">
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        <div className="settings-modal-tabs" role="tablist">
          {[
            { id: 'all', label: 'All Stages', icon: 'apps' },
            { id: 'voices', label: 'Voices', icon: 'manage_accounts' },
            { id: 'LLM & Translation', label: 'LLM & Trans', icon: 'psychology' },
            { id: 'Transcription & Diarization', label: 'Transcription', icon: 'subtitles' },
            { id: 'Voice Generation (TTS)', label: 'TTS Engine', icon: 'record_voice_over' },
            { id: 'Timing & Mixing', label: 'Timing & Mix', icon: 'tune' },
            { id: 'Emotions & Context', label: 'Emotions', icon: 'mood' },
            { id: 'Debug & Watermarks', label: 'Debug', icon: 'bug_report' },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              className={`settings-tab-btn${activeTab === tab.id ? ' active' : ''}`}
              onClick={() => onTabChange(tab.id)}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        <div className="settings-modal-body">
          {(activeTab === 'all' || activeTab === 'voices') && (
            <div className="settings-group-card settings-group-card--voice">
              <div className="settings-group-title">
                <span className="material-symbols-outlined">manage_accounts</span>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flex: 1 }}>
                  <span>Voice Assignment</span>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'none', fontWeight: 500 }}>
                    {displaySpeakers.length} Detected Speakers
                  </span>
                </div>
              </div>
              <p className="settings-section-desc" style={{ marginBottom: '1rem' }}>
                Map each detected speaker to a voice profile from the global library.
              </p>

              {displaySpeakers.length === 0 ? (
                <div className="settings-empty">
                  <span className="material-symbols-outlined" style={{ fontSize: '28px', color: 'var(--text-muted)' }}>person_off</span>
                  <span>No speakers detected. Open a project first.</span>
                </div>
              ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.75rem' }}>
                  {displaySpeakers.map((speaker, idx) => (
                    <div
                      key={speaker}
                      style={{
                        background: 'var(--bg-subtle, #1c1b1b)',
                        border: '1px solid rgba(255,255,255,0.06)',
                        borderRadius: '8px',
                        padding: '0.625rem 0.875rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.75rem',
                      }}
                    >
                      <span className={`speaker-dot speaker-dot-${idx % 4}`} style={{ width: '8px', height: '8px', borderRadius: '50%', flexShrink: 0 }} />
                      <span className="voice-assignment-speaker" style={{ fontWeight: 600, minWidth: '85px' }}>{speaker}</span>
                      <select
                        className="voice-assignment-select"
                        style={{ flex: 1, padding: '0.35rem 0.5rem', fontSize: '0.8125rem' }}
                        value={speakerMap[speaker] ?? ''}
                        onChange={(e) => onSpeakerMapChange({ ...speakerMap, [speaker]: e.target.value })}
                        aria-label={`Voice profile for ${speaker}`}
                      >
                        <option value="">— no override —</option>
                        {Object.keys(voiceProfiles).map((name) => (
                          <option key={name} value={name}>{name}</option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>
              )}

              {speakerMapStatus && (
                <div className={`settings-status-msg${speakerMapStatus.includes('fail') || speakerMapStatus.includes('Open') ? ' settings-status-msg--err' : ''}`} style={{ marginTop: '0.75rem' }}>
                  {speakerMapStatus}
                </div>
              )}
            </div>
          )}

          {globalConfig && (() => {
            const grouped = globalConfig.schema.fields.reduce((acc, field) => {
              if (field.type === 'list' || field.type === 'object' || field.type === 'structured') return acc;
              const sec = field.section || getFieldSection(field.name);
              if (!acc[sec]) acc[sec] = [];
              acc[sec].push(field);
              return acc;
            }, {} as Record<string, SchemaFieldDefinition[]>);

            const filteredGroups = Object.entries(grouped).filter(([secName]) => {
              if (activeTab === 'all') return true;
              return activeTab === secName;
            });

            return (
              <div style={{ display: 'grid', gridTemplateColumns: filteredGroups.length === 1 ? '1fr' : 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1rem' }}>
                {filteredGroups.map(([secName, fields]) => {
                  let icon = 'settings_input_component';
                  let cardClass = 'settings-group-card--general';
                  if (secName.includes('LLM') || secName.includes('Translation')) { icon = 'psychology'; cardClass = 'settings-group-card--llm'; }
                  else if (secName.includes('Transcription')) { icon = 'subtitles'; cardClass = 'settings-group-card--transcription'; }
                  else if (secName.includes('Voice')) { icon = 'record_voice_over'; cardClass = 'settings-group-card--voice'; }
                  else if (secName.includes('Emotion')) { icon = 'mood'; cardClass = 'settings-group-card--emotion'; }
                  else if (secName.includes('Timing') || secName.includes('Mixing')) { icon = 'tune'; cardClass = 'settings-group-card--timing'; }
                  else if (secName.includes('Debug')) { icon = 'bug_report'; cardClass = 'settings-group-card--debug'; }
                  else if (secName.includes('Execution')) { icon = 'power_settings_new'; cardClass = 'settings-group-card--general'; }

                  return (
                    <div key={secName} className={`settings-group-card ${cardClass}`}>
                      <div className="settings-group-title">
                        <span className="material-symbols-outlined">{icon}</span>
                        <span>{secName}</span>
                      </div>
                      <div className="settings-field-flow">
                        {fields.map(field => (
                          <SchemaField
                            key={field.name}
                            field={field}
                            value={globalSettingsValues[field.name]}
                            options={normalizeOptions(field.options ?? globalOptions[field.options_key ?? field.name])}
                            onChange={(value) => onGlobalSettingsValuesChange({ ...globalSettingsValues, [field.name]: value })}
                          />
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })()}
        </div>

        <div className="settings-modal-footer">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {globalSettingsStatus && (
              <span className={`settings-status-msg${globalSettingsStatus.includes('fail') ? ' settings-status-msg--err' : ''}`} style={{ margin: 0 }}>
                {globalSettingsStatus}
              </span>
            )}
          </div>
          <div className="settings-modal-footer-actions">
            <button type="button" className="btn-secondary" onClick={onClose} style={{ padding: '0.45rem 1rem' }}>
              Close
            </button>
            <button
              type="button"
              className="btn-primary"
              onClick={() => {
                if (displaySpeakers.length > 0) {
                  onSaveSpeakerMap();
                }
                if (globalConfig) {
                  onSaveGlobalSettings();
                }
              }}
              disabled={globalSettingsSaving || speakerMapSaving}
              style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', padding: '0.45rem 1.25rem' }}
            >
              {globalSettingsSaving || speakerMapSaving
                ? <><span className="material-symbols-outlined spin-anim" style={{ fontSize: '16px' }}>progress_activity</span> Applying…</>
                : <><span className="material-symbols-outlined" style={{ fontSize: '16px' }}>bolt</span> Save & Apply</>}
            </button>
          </div>
        </div>
      </div>
    </>
  )
}
