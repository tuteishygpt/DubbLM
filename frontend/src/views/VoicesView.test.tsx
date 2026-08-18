import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { ApiClient } from '../api/types'
import { VoicesView } from './VoicesView'

function api(): ApiClient {
  return {
    get: vi.fn(async (path: string) => path === '/api/options' ? {
      tts_providers: ['gemini', 'openai', 'higgs'],
      tts_models: { gemini: ['model-a'], openai: ['model-b'], higgs: [] },
      tts_voices: { gemini: ['voice-a'], openai: ['voice-b'], higgs: [] },
      tts_reference_capabilities: { gemini: 'unsupported', openai: 'unsupported', higgs: 'required' },
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
  it('creates a new voice profile and saves it', async () => {
    const user = userEvent.setup()
    const client = api()
    render(<VoicesView client={client} />)
    expect(await screen.findByLabelText(/Profile name/i)).toHaveValue('SPEAKER_00')

    await user.click(screen.getAllByRole('button', { name: /New profile/i })[0])
    expect(screen.getByLabelText(/Profile name/i)).toHaveValue('SPEAKER_01')

    await user.clear(screen.getByLabelText(/Profile name/i))
    await user.type(screen.getByLabelText(/Profile name/i), 'John Male')
    await user.selectOptions(screen.getByLabelText('Provider'), 'openai')
    await user.selectOptions(screen.getByLabelText('Model'), 'model-b')
    await user.selectOptions(screen.getByLabelText('Voice'), 'voice-b')
    await user.click(screen.getByRole('button', { name: /Save profile/i }))

    expect(client.put).toHaveBeenCalledWith('/api/voice-profiles/John%20Male', {
      revision: 'settings-1',
      profile: { tts_system: 'openai', model: 'model-b', voice_name: 'voice-b' },
    })
  })

  it('saves voice cloning profile with explicit reference_mode', async () => {
    const user = userEvent.setup()
    const client = api()
    render(<VoicesView client={client} />)
    expect(await screen.findByLabelText(/Profile name/i)).toHaveValue('SPEAKER_00')

    await user.click(screen.getAllByRole('button', { name: /New profile/i })[0])
    expect(screen.getByLabelText(/Profile name/i)).toHaveValue('SPEAKER_01')

    await user.selectOptions(screen.getByLabelText('Provider'), 'higgs')
    expect(screen.getByLabelText('Reference Mode')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Reference Mode'), 'speaker')
    await user.click(screen.getByRole('button', { name: /Save profile/i }))

    expect(client.put).toHaveBeenCalledWith('/api/voice-profiles/SPEAKER_01', {
      revision: 'settings-1',
      profile: { tts_system: 'higgs', reference_mode: 'speaker' },
    })
  })

  it('renders reference audio options by reference name and saves configured reference audio', async () => {
    const user = userEvent.setup()
    const client = api()
    render(<VoicesView client={client} />)
    expect(await screen.findByLabelText(/Profile name/i)).toHaveValue('SPEAKER_00')

    await user.selectOptions(screen.getByLabelText('Provider'), 'higgs')
    expect(screen.getByLabelText('Reference Mode')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Reference Mode'), 'configured')

    const refSelect = screen.getByLabelText('Reference Audio')
    expect(refSelect).toBeInTheDocument()
    // Should display Reference Name ('Narrator'), not filename ('sample.wav')
    expect(screen.getByRole('option', { name: 'Narrator' })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: 'sample.wav' })).not.toBeInTheDocument()

    await user.selectOptions(refSelect, 'Narrator')
    await user.click(screen.getByRole('button', { name: /Save profile/i }))

    expect(client.put).toHaveBeenCalledWith('/api/voice-profiles/SPEAKER_00', {
      revision: 'settings-1',
      profile: { tts_system: 'higgs', reference_mode: 'configured', reference_audio: 'Narrator' },
    })
  })
})



