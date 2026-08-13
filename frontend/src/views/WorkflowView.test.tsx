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
}

function client(): ApiClient {
  return {
    get: vi.fn(async (path: string) => path === '/api/options' ? options : {}) as ApiClient['get'],
    put: vi.fn(),
    post: vi.fn().mockResolvedValue({ id: 'job-8', status: 'queued' }),
    delete: vi.fn(),
    upload: vi.fn()
      .mockResolvedValueOnce({ id: 'video-1' })
      .mockResolvedValueOnce({ id: 'track-1' }),
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

    await user.click(screen.getByRole('button', { name: 'Queue job' }))
    expect(screen.getByRole('alert')).toHaveTextContent('Video is required')

    await user.upload(screen.getByLabelText('Video'), new File(['video'], 'movie.mp4', { type: 'video/mp4' }))
    await user.upload(screen.getByLabelText('Isolated audio track'), new File(['audio'], 'voice.wav', { type: 'audio/wav' }))
    await user.selectOptions(screen.getByLabelText('Target language'), 'es')
    await user.selectOptions(screen.getByLabelText('Speaker report mode'), 'detailed')
    await user.click(screen.getByLabelText('Include subtitles'))
    await user.selectOptions(screen.getByLabelText('Run mode'), 'preview')
    fireEvent.change(screen.getByLabelText('Speaker-label mapping'), { target: { value: JSON.stringify({ SPEAKER_00: 'voice.wav' }) } })
    await user.click(screen.getByRole('button', { name: 'Queue job' }))

    await waitFor(() => expect(api.post).toHaveBeenCalledWith('/api/jobs', {
      input_upload_id: 'video-1',
      isolated_tracks: { SPEAKER_00: 'track-1' },
      overrides: { target_language: 'es', speaker_report_mode: 'detailed', include_subtitles: true, run_mode: 'preview' },
    }))
    expect(api.upload).toHaveBeenNthCalledWith(1, '/api/uploads', expect.any(FormData))
    expect(api.upload).toHaveBeenNthCalledWith(2, '/api/uploads', expect.any(FormData))
    expect(screen.getByRole('status')).toHaveTextContent('Job job-8 queued')
  })
})
