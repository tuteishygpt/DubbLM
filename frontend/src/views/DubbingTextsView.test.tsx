import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ApiRequestError } from '../api/client'
import type { ApiClient } from '../api/types'
import { DubbingTextsView } from './DubbingTextsView'

function api(conflict = false): ApiClient {
  return {
    get: vi.fn(async (path: string) => path === '/api/jobs' ? { jobs: [{ id: 'job-1', status: 'completed' }, { id: 'job-2', status: 'running' }] } : {
      revision: 'texts-7', source: 'seeded', cached: true, segments: [
        { id: 'seg-stable-1', speaker: 'SPEAKER_00', start: 1.2, end: 2.8, original_text: 'Hello', translated_text: 'Bonjour', synthesized_text: 'Bonjour!', audio_path: 'cache/a.wav', audio_url: '/api/audio/a.wav' },
        { id: 'seg-stable-2', speaker: 'SPEAKER_01', start: 3, end: 4, original_text: 'Bye', translated_text: 'Au revoir', synthesized_text: '', audio_path: null, audio_url: null },
      ],
    }) as ApiClient['get'],
    put: conflict ? vi.fn().mockRejectedValue(new ApiRequestError('Text revision conflict', 'revision_conflict', 409)) : vi.fn().mockResolvedValue({ revision: 'texts-8' }),
    post: vi.fn().mockResolvedValue({ segment: { id: 'seg-stable-1', synthesized_text: 'Override speech', audio_path: 'cache/new.wav', audio_url: '/api/audio/new.wav' } }),
    delete: vi.fn(), upload: vi.fn(), subscribeJobEvents: vi.fn(() => () => undefined),
  }
}

describe('DubbingTextsView', () => {
  it('loads seeded/cached segments, preserves ids, edits columns, saves, and regenerates only the selected row', async () => {
    const user = userEvent.setup(); const client = api(); render(<DubbingTextsView client={client} />)
    await user.selectOptions(await screen.findByLabelText('Job'), 'job-1')
    expect(await screen.findByText('Seeded · cached')).toBeInTheDocument()
    const first = screen.getByTestId('segment-seg-stable-1'); const second = screen.getByTestId('segment-seg-stable-2')
    for (const label of ['Speaker seg-stable-1', 'Original text seg-stable-1', 'Translated text seg-stable-1', 'Synthesized text seg-stable-1']) expect(screen.getByLabelText(label)).toBeInTheDocument()
    expect(within(first).getByText('1.2–2.8')).toBeInTheDocument(); expect(within(first).getByText('cache/a.wav')).toBeInTheDocument()
    expect(within(first).getByRole('link', { name: 'Play audio' })).toHaveAttribute('href', '/api/audio/a.wav')
    expect(within(second).getByText('Missing audio')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Save changes' })).toBeDisabled()
    await user.clear(screen.getByLabelText('Translated text seg-stable-1')); await user.type(screen.getByLabelText('Translated text seg-stable-1'), 'Salut')
    expect(screen.getByText('Unsaved changes')).toBeInTheDocument(); expect(screen.getByRole('button', { name: 'Save changes' })).toBeEnabled()
    await user.click(screen.getByLabelText('Select seg-stable-1')); await user.clear(screen.getByLabelText('Synthesized text seg-stable-1')); await user.type(screen.getByLabelText('Synthesized text seg-stable-1'), 'Override speech')
    await user.click(screen.getByRole('button', { name: 'Regenerate selected' }))
    expect(client.post).toHaveBeenCalledWith('/api/jobs/job-1/dubbing-texts/seg-stable-1/regenerate', { synthesized_text: 'Override speech' })
    await user.click(screen.getByRole('button', { name: 'Save changes' }))
    await waitFor(() => expect(client.put).toHaveBeenCalledWith('/api/jobs/job-1/dubbing-texts', { revision: 'texts-7', segments: expect.arrayContaining([expect.objectContaining({ id: 'seg-stable-1', translated_text: 'Salut' }), expect.objectContaining({ id: 'seg-stable-2' })]) }))
  })

  it('surfaces save revision conflicts', async () => {
    const user = userEvent.setup(); render(<DubbingTextsView client={api(true)} />); await user.selectOptions(await screen.findByLabelText('Job'), 'job-1'); await screen.findByTestId('segment-seg-stable-1')
    await user.type(screen.getByLabelText('Translated text seg-stable-1'), '!'); await user.click(screen.getByRole('button', { name: 'Save changes' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Text revision conflict')
  })
})
