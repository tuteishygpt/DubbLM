import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { ApiClient } from '../api/types'
import { VoicesView } from './VoicesView'

function api(): ApiClient {
  return {
    get: vi.fn(async (path: string) => path === '/api/options' ? {
      tts_providers: ['gemini', 'openai'],
      tts_models: { gemini: ['model-a'], openai: ['model-b'] },
      tts_voices: { gemini: ['voice-a'], openai: ['voice-b'] },
    } : path === '/api/voice-profiles' ? {
      revision: 'settings-1',
      profiles: { SPEAKER_00: { tts_system: 'gemini', model: 'model-a', voice_name: 'voice-a' } },
    } : {
      revision: 'refs-1',
      entries: [{ speaker_id: 'Narrator', reference_text: 'Warm delivery', audio: { id: 'ref-1', name: 'sample.wav', url: '/api/reference-library/Narrator/audio' } }],
    }) as ApiClient['get'],
    put: vi.fn(async (path: string) => path.startsWith('/api/voice-profiles/') ? {
      revision: 'settings-2', profiles: { SPEAKER_00: { tts_system: 'openai', model: 'model-b', voice_name: 'voice-b' } },
    } : { settings_revision: 'settings-3' }) as ApiClient['put'],
    post: vi.fn(),
    delete: vi.fn(async (path: string) => path.startsWith('/api/voice-profiles/') ? { revision: 'settings-3', profiles: {} } : { revision: 'refs-3', entries: [] }) as ApiClient['delete'],
    upload: vi.fn().mockResolvedValue({ revision: 'refs-2', entries: [
      { speaker_id: 'Narrator', reference_text: 'Warm delivery', audio: { id: 'ref-1', name: 'sample.wav', url: '/api/reference-library/Narrator/audio' } },
      { speaker_id: 'SPEAKER_00', reference_text: '', audio: { id: 'ref-2', name: 'new.wav', url: '/api/reference-library/SPEAKER_00/audio' } },
    ] }),
    subscribeJobEvents: vi.fn(() => () => undefined),
  }
}

describe('VoicesView', () => {
  it('uses revisioned profile and reference-library contracts', async () => {
    const user = userEvent.setup(); const client = api(); render(<VoicesView client={client} />)
    expect(await screen.findByLabelText('Name')).toHaveValue('SPEAKER_00')
    expect(screen.getByRole('option', { name: 'voice-a' })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: 'voice-b' })).not.toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Provider'), 'openai')
    expect(screen.getByRole('option', { name: 'voice-b' })).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Model'), 'model-b')
    await user.selectOptions(screen.getByLabelText('Voice'), 'voice-b')
    await user.click(screen.getByRole('button', { name: 'Save profile' }))
    expect(client.put).toHaveBeenCalledWith('/api/voice-profiles/SPEAKER_00', {
      revision: 'settings-1',
      profile: { tts_system: 'openai', model: 'model-b', voice_name: 'voice-b' },
    })

    await user.type(screen.getByLabelText('Speaker label'), 'SPEAKER_00')
    await user.upload(screen.getByLabelText('Reference audio'), new File(['x'], 'new.wav', { type: 'audio/wav' }))
    await user.click(screen.getByRole('button', { name: 'Upload reference' }))
    await waitFor(() => expect(client.upload).toHaveBeenCalledWith('/api/reference-library', expect.any(FormData)))
    const uploadForm = vi.mocked(client.upload).mock.calls[0][1]
    expect(uploadForm.get('speaker_id')).toBe('SPEAKER_00')
    expect(uploadForm.get('revision')).toBe('refs-1')

    await user.selectOptions(screen.getByLabelText('Reference'), 'SPEAKER_00')
    await user.click(screen.getByRole('button', { name: 'Assign profile' }))
    expect(client.put).toHaveBeenCalledWith('/api/reference-library/SPEAKER_00', {
      profile_speaker_id: 'SPEAKER_00', settings_revision: 'settings-2',
    })

    await user.click(screen.getByRole('button', { name: 'Delete sample.wav' }))
    expect(client.delete).toHaveBeenCalledWith('/api/reference-library/Narrator', { revision: 'refs-2' })

    await user.click(screen.getByRole('button', { name: 'Delete profile' }))
    expect(client.delete).toHaveBeenCalledWith('/api/voice-profiles/SPEAKER_00', { revision: 'settings-3' })
  })
})
