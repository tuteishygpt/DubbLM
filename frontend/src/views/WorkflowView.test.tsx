import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { ApiClient, ConfigResponse, OptionsResponse } from '../api/types'
import { WorkflowView } from './WorkflowView'

const config: ConfigResponse = {
  revision: 'config-3',
  values: {
    source_language: 'en', target_language: 'fr', speaker_report_mode: 'summary',
    include_subtitles: false, audio_only: false, quality: 'balanced', run_mode: 'full',
  },
  schema: {
    fields: [
      { name: 'source_language', label: 'Source language', type: 'select', scope: 'workflow', options_key: 'languages' },
      { name: 'target_language', label: 'Target language', type: 'select', workflow: true, options_key: 'languages' },
      { name: 'speaker_report_mode', label: 'Speaker report mode', type: 'select', workflow: true, options_key: 'report_modes' },
      { name: 'include_subtitles', label: 'Include subtitles', type: 'boolean', workflow: true },
      { name: 'audio_only', label: 'Audio only', type: 'boolean', workflow: true },
      { name: 'quality', label: 'Quality', type: 'select', workflow: true, options_key: 'qualities' },
      { name: 'run_mode', label: 'Run mode', type: 'select', workflow: true, options_key: 'run_modes' },
      { name: 'internal_key', label: 'Internal key', type: 'text', scope: 'settings' },
    ],
  },
}

const options: OptionsResponse = {
  languages: [{ value: 'en', label: 'English' }, { value: 'fr', label: 'French' }, { value: 'es', label: 'Spanish' }],
  report_modes: [{ value: 'summary', label: 'Summary' }, { value: 'detailed', label: 'Detailed' }],
  qualities: [{ value: 'balanced', label: 'Balanced' }, { value: 'studio', label: 'Studio' }],
  run_modes: [{ value: 'full', label: 'Full pipeline' }, { value: 'preview', label: 'Preview' }],
  tts_providers: ['gemini', 'openai', 'omnivoice'],
  tts_models: { gemini: ['gemini-2.5-pro-preview-tts'], openai: ['tts-1', 'tts-1-hd'] },
  tts_voices: { gemini: ['Kore', 'Puck'], openai: ['alloy', 'nova'] },
}

function client(): ApiClient {
  return {
    get: vi.fn(async (path: string) => {
      if (path === '/api/options') return options
      if (path === '/api/voice-profiles') return { revision: 'v1', profiles: { Narrator: { tts_system: 'gemini', voice_name: 'Puck' } } }
      if (path === '/api/reference-library') return { revision: 'r1', entries: [{ speaker_id: 'ref1', reference_text: 'Sample', audio: { id: 'a1', name: 'sample.wav', url: '/sample.wav' } }] }
      return {}
    }) as ApiClient['get'],
    put: vi.fn(),
    post: vi.fn().mockResolvedValue({ id: 'job-8', status: 'queued' }),
    delete: vi.fn(),
    upload: vi.fn()
      .mockResolvedValueOnce({ id: 'video-1' })
      .mockResolvedValueOnce({ id: 'track-1' })
      .mockResolvedValueOnce({ id: 'track-2' }),
    subscribeJobEvents: vi.fn(() => () => undefined),
  }
}

describe('WorkflowView', () => {
  it('renders the complete schema-driven workflow without client-side inventories', async () => {
    const api = client()
    render(<WorkflowView client={api} config={config} />)

    expect((await screen.findAllByRole('option', { name: 'Spanish' })).length).toBe(2)
    for (const label of [
      'Video', 'Isolated audio track', 'Speaker-label mapping', 'Source language', 'Target language',
      'Speaker report mode', 'Include subtitles', 'Audio only', 'Quality', 'Run mode',
    ]) expect(screen.getByLabelText(label)).toBeInTheDocument()
    expect(screen.queryByLabelText('Internal key')).not.toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Preview' })).toBeInTheDocument()
  })

  it('validates input, uploads media, and submits only workflow overrides', async () => {
    const user = userEvent.setup()
    const api = client()
    render(<WorkflowView client={api} config={config} />)
    await screen.findAllByRole('option', { name: 'Spanish' })

    await user.click(screen.getByRole('button', { name: 'Start' }))
    expect(screen.getByRole('alert')).toHaveTextContent('Video is required')

    await user.upload(screen.getByLabelText('Video'), new File(['video'], 'movie.mp4', { type: 'video/mp4' }))
    await user.upload(screen.getByLabelText('Isolated audio track'), new File(['audio'], 'voice.wav', { type: 'audio/wav' }))
    await user.selectOptions(screen.getByLabelText('Target language'), 'es')
    await user.selectOptions(screen.getByLabelText('Speaker report mode'), 'detailed')
    await user.click(screen.getByLabelText('Include subtitles'))
    await user.selectOptions(screen.getByLabelText('Run mode'), 'preview')
    fireEvent.change(screen.getByLabelText('Speaker-label mapping'), { target: { value: JSON.stringify({ SPEAKER_00: 'voice.wav' }) } })

    // Disable voice overrides for standard submission
    await user.click(screen.getByLabelText('Configure Dubbing Voices'))
    await user.click(screen.getByRole('button', { name: 'Start' }))

    await waitFor(() => expect(api.post).toHaveBeenCalledWith('/api/jobs', {
      input_upload_id: 'video-1',
      isolated_tracks: { SPEAKER_00: 'track-1' },
      overrides: { target_language: 'es', speaker_report_mode: 'detailed', include_subtitles: true, run_mode: 'preview' },
    }))
    expect(api.upload).toHaveBeenNthCalledWith(1, '/api/uploads', expect.any(FormData))
    expect(api.upload).toHaveBeenNthCalledWith(2, '/api/uploads', expect.any(FormData))
    expect(screen.getByRole('status')).toHaveTextContent('Job job-8 queued')
  })

  it('supports visual speaker track cards and custom voice profile configuration', async () => {
    const user = userEvent.setup()
    const api = client()
    render(<WorkflowView client={api} config={config} />)
    await screen.findAllByRole('option', { name: 'Spanish' })

    await user.upload(screen.getByLabelText('Video'), new File(['video'], 'movie.mp4', { type: 'video/mp4' }))
    await user.upload(screen.getByLabelText('Isolated audio track'), [
      new File(['audio1'], 'speaker_00.wav', { type: 'audio/wav' }),
      new File(['audio2'], 'speaker_01.wav', { type: 'audio/wav' }),
    ])

    expect(screen.getByText(/Isolated Audio Tracks & Voice Assignment \(2\)/i)).toBeInTheDocument()
    expect(screen.getByText('speaker_00.wav')).toBeInTheDocument()
    expect(screen.getByText('speaker_01.wav')).toBeInTheDocument()

    // Configure Voice Settings for Default and SPEAKER_01
    expect(screen.getByText('Dubbing Voice Selection')).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /Default Voice \(\*\)/i })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /SPEAKER_00/i })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /SPEAKER_01/i })).toBeInTheDocument()

    // Switch to SPEAKER_01 tab, expand advanced parameters and select provider
    await user.click(screen.getByRole('tab', { name: /SPEAKER_01/i }))
    await user.click(screen.getByText(/Customize detailed parameters/i))
    await user.selectOptions(screen.getByLabelText('TTS Provider'), 'openai')
    await user.type(screen.getByLabelText(/Style Prompt/i), 'warm voice')

    await user.click(screen.getByRole('button', { name: 'Start' }))

    await waitFor(() => expect(api.post).toHaveBeenCalledWith('/api/jobs', expect.objectContaining({
      input_upload_id: 'video-1',
      isolated_tracks: {
        SPEAKER_00: 'track-1',
        SPEAKER_01: 'track-2',
      },
      overrides: expect.objectContaining({
        voices: expect.objectContaining({
          '*': expect.objectContaining({ tts_system: 'gemini' }),
          SPEAKER_01: expect.objectContaining({ tts_system: 'openai', style_prompt: 'warm voice' }),
        }),
      }),
    })))
  })

  it('prevents duplicate submission while starting the selected dubbing job', async () => {
    const user = userEvent.setup()
    let finishUpload: ((value: { id: string }) => void) | undefined
    const api = client()
    api.upload = vi.fn((_path: string, _data: FormData) => new Promise<unknown>((resolve) => {
      finishUpload = resolve
    })) as unknown as ApiClient['upload']
    render(<WorkflowView client={api} config={config} />)
    await screen.findAllByRole('option', { name: 'Spanish' })

    await user.upload(screen.getByLabelText('Video'), new File(['video'], 'movie.mp4', { type: 'video/mp4' }))
    await user.click(screen.getByRole('button', { name: 'Start' }))

    expect(screen.getByRole('button', { name: 'Starting…' })).toBeDisabled()
    expect(api.upload).toHaveBeenCalledTimes(1)

    finishUpload?.({ id: 'video-1' })
    await waitFor(() => expect(api.post).toHaveBeenCalledTimes(1))
  })
})
