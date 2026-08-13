import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { ApiClient } from '../api/types'
import { VoicesView } from './VoicesView'

function api(): ApiClient {
  return {
    get: vi.fn(async (path: string) => path === '/api/options' ? {
      voice_models: [{ value: 'model-a', label: 'Model A' }, { value: 'model-b', label: 'Model B' }],
      voices: [{ value: 'voice-a', label: 'Voice A', depends_on: { model: 'model-a' } }, { value: 'voice-b', label: 'Voice B', depends_on: { model: 'model-b' } }],
    } : { profiles: [{ id: 'profile-1', name: 'Narrator', model: 'model-a', voice: 'voice-a', reference_id: 'ref-1' }], references: [{ id: 'ref-1', name: 'sample.wav' }] }) as ApiClient['get'],
    put: vi.fn().mockResolvedValue({}), post: vi.fn().mockResolvedValue({}), delete: vi.fn().mockResolvedValue(undefined),
    upload: vi.fn().mockResolvedValue({ id: 'ref-2', name: 'new.wav' }), subscribeJobEvents: vi.fn(() => () => undefined),
  }
}

describe('VoicesView', () => {
  it('filters dependent choices and supports profile upsert, deletion, assignment, and references', async () => {
    const user = userEvent.setup(); const client = api(); render(<VoicesView client={client} />)
    expect(await screen.findByLabelText('Name')).toHaveValue('Narrator')
    expect(screen.getByRole('option', { name: 'Voice A' })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: 'Voice B' })).not.toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Model'), 'model-b')
    expect(screen.getByRole('option', { name: 'Voice B' })).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Voice'), 'voice-b')
    await user.selectOptions(screen.getByLabelText('Reference'), 'ref-1')
    await user.click(screen.getByRole('button', { name: 'Save profile' }))
    expect(client.put).toHaveBeenCalledWith('/api/voice-profiles/profile-1', expect.objectContaining({ model: 'model-b', voice: 'voice-b', reference_id: 'ref-1' }))
    await user.click(screen.getByRole('button', { name: 'Delete profile' })); expect(client.delete).toHaveBeenCalledWith('/api/voice-profiles/profile-1')
    await user.upload(screen.getByLabelText('Reference audio'), new File(['x'], 'new.wav', { type: 'audio/wav' })); await user.click(screen.getByRole('button', { name: 'Upload reference' }))
    await waitFor(() => expect(client.upload).toHaveBeenCalledWith('/api/references', expect.any(FormData)))
    await user.click(screen.getByRole('button', { name: 'Delete sample.wav' })); expect(client.delete).toHaveBeenCalledWith('/api/references/ref-1')
    await user.type(screen.getByLabelText('Speaker label'), 'SPEAKER_00'); await user.click(screen.getByRole('button', { name: 'Assign profile' }))
    expect(client.put).toHaveBeenCalledWith('/api/voice-assignments/SPEAKER_00', { profile_id: 'profile-1' })
  })
})
