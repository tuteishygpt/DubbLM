import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ApiRequestError } from '../api/client'
import type { ApiClient } from '../api/types'
import { DubbingTextsView } from './DubbingTextsView'

const segment = {
  segment_id: 'seg-stable-1', speaker: 'SPEAKER_00', start: 1.2, end: 2.8,
  text: 'Hello', translation: 'Bonjour', synthesized_text: 'Bonjour!', style_prompt: 'Warmly',
  audio: { id: 'audio-1', name: 'a.wav', url: '/api/jobs/job-1/files/audio-1' },
}

function api(conflict = false): ApiClient {
  return {
    get: vi.fn(async (path: string) => path === '/api/jobs' ? { jobs: [{ id: 'job-1', status: 'succeeded' }, { id: 'job-2', status: 'running' }] } : {
      revision: 'texts-7', source: 'snapshot', segments: [segment, {
        segment_id: 'seg-stable-2', speaker: 'SPEAKER_01', start: 3, end: 4,
        text: 'Bye', translation: 'Au revoir', synthesized_text: '', style_prompt: '', audio: null,
      }],
    }) as ApiClient['get'],
    put: conflict ? vi.fn().mockRejectedValue(new ApiRequestError('Text revision conflict', 'revision_conflict', 409)) : vi.fn().mockResolvedValue({ revision: 'texts-8', source: 'snapshot', segments: [] }),
    post: vi.fn().mockResolvedValue({ revision: 'texts-9', segment: { ...segment, synthesized_text: 'Override speech', audio: { id: 'audio-2', name: 'new.wav', url: '/api/jobs/job-1/files/audio-2' } } }),
    delete: vi.fn(), upload: vi.fn(), subscribeJobEvents: vi.fn(() => () => undefined),
  }
}

describe('DubbingTextsView', () => {
  it('loads backend segments, saves exact editable fields, and regenerates with the revision', async () => {
    const user = userEvent.setup(); const client = api(); render(<DubbingTextsView client={client} />)
    await user.selectOptions(await screen.findByLabelText('Job'), 'job-1')
    expect(await screen.findByText('Snapshot')).toBeInTheDocument()
    const first = screen.getByTestId('segment-seg-stable-1'); const second = screen.getByTestId('segment-seg-stable-2')
    for (const label of ['Speaker seg-stable-1', 'Original text seg-stable-1', 'Translated text seg-stable-1', 'Synthesized text seg-stable-1', 'Style instructions seg-stable-1']) expect(screen.getByLabelText(label)).toBeInTheDocument()
    expect(within(first).getByText('1.2–2.8')).toBeInTheDocument()
    expect(within(first).getByText('a.wav')).toBeInTheDocument()
    expect(within(first).getByRole('link', { name: 'Play audio' })).toHaveAttribute('href', '/api/jobs/job-1/files/audio-1')
    expect(within(second).getByText('Missing audio')).toBeInTheDocument()

    await user.clear(screen.getByLabelText('Translated text seg-stable-1')); await user.type(screen.getByLabelText('Translated text seg-stable-1'), 'Salut')
    await user.click(screen.getByLabelText('Select seg-stable-1'))
    await user.clear(screen.getByLabelText('Synthesized text seg-stable-1')); await user.type(screen.getByLabelText('Synthesized text seg-stable-1'), 'Override speech')
    await user.click(screen.getByRole('button', { name: 'Regenerate selected' }))
    expect(client.post).toHaveBeenCalledWith('/api/jobs/job-1/dubbing-texts/seg-stable-1/regenerate', { revision: 'texts-7', synthesized_text: 'Override speech' })
    expect(within(first).getByText('new.wav')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Save changes' }))
    await waitFor(() => expect(client.put).toHaveBeenCalledWith('/api/jobs/job-1/dubbing-texts', {
      revision: 'texts-9',
      segments: expect.arrayContaining([
        { segment_id: 'seg-stable-1', speaker: 'SPEAKER_00', start: 1.2, end: 2.8, text: 'Hello', translation: 'Salut', synthesized_text: 'Override speech', style_prompt: 'Warmly' },
        { segment_id: 'seg-stable-2', speaker: 'SPEAKER_01', start: 3, end: 4, text: 'Bye', translation: 'Au revoir', synthesized_text: '', style_prompt: '' },
      ]),
    }))
  })

  it('surfaces save revision conflicts', async () => {
    const user = userEvent.setup(); render(<DubbingTextsView client={api(true)} />); await user.selectOptions(await screen.findByLabelText('Job'), 'job-1'); await screen.findByTestId('segment-seg-stable-1')
    await user.type(screen.getByLabelText('Translated text seg-stable-1'), '!'); await user.click(screen.getByRole('button', { name: 'Save changes' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Text revision conflict')
  })
})
