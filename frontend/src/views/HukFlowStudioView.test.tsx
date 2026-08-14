import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { ApiClient } from '../api/types'
import { HukFlowStudioView } from './HukFlowStudioView'

const segment1 = {
  segment_id: 'seg-a1', speaker: 'SPEAKER_00', start: 1.5, end: 4.0,
  text: 'The rapid advancement of these models has completely shifted our paradigm.',
  translation: 'Хуткае развіццё гэтых мадэляў цалкам змяніла нашу парадыгму.',
  synthesized_text: 'Хуткае развіццё…', style_prompt: 'Neutral',
  audio: { id: 'aud-1', name: 'seg-a1.wav', url: '/api/jobs/job-1/files/aud-1' },
}

const segment2 = {
  segment_id: 'seg-a2', speaker: 'SPEAKER_01', start: 18.0, end: 20.5,
  text: 'We need to consider the ethical implications before deployment.',
  translation: 'Мы павінны ўлічваць этычныя наступствы перад разгортваннем.',
  synthesized_text: '', style_prompt: '',
  audio: null,
}

const textDocument = {
  revision: 'rev-studio-1',
  source: 'snapshot',
  segments: [segment1, segment2],
}

function makeClient(overrides: Partial<ApiClient> = {}): ApiClient {
  return {
    get: vi.fn(async (path: string) => {
      if (path === '/api/jobs') return { jobs: [{ id: 'job-1', status: 'succeeded' }] }
      if (path.includes('/dubbing-texts')) return textDocument
      return {}
    }) as ApiClient['get'],
    post: vi.fn().mockResolvedValue({
      revision: 'rev-studio-2',
      segment: { ...segment2, synthesized_text: 'Новы пераклад', audio: { id: 'aud-2', name: 'new.wav', url: '/api/jobs/job-1/files/aud-2' } },
    }),
    put: vi.fn().mockResolvedValue({ ...textDocument, revision: 'rev-studio-3' }),
    delete: vi.fn(),
    upload: vi.fn(),
    subscribeJobEvents: vi.fn(() => () => undefined),
    ...overrides,
  }
}

describe('HukFlowStudioView', () => {
  it('loads real dubbing-texts, renders segment cards, edits text, and saves', async () => {
    const user = userEvent.setup()
    const client = makeClient()
    render(<HukFlowStudioView client={client} />)

    // Heading exists for a11y
    expect(screen.getByRole('heading', { name: 'HukFlow Studio', level: 1 })).toBeInTheDocument()

    // Waits for segments to load
    expect(await screen.findByText('SPEAKER_00')).toBeInTheDocument()
    expect(screen.getByText('SPEAKER_01')).toBeInTheDocument()

    // Jobs API called first
    expect(client.get).toHaveBeenCalledWith('/api/jobs')

    // Dubbing texts API called with the first job
    await waitFor(() =>
      expect(client.get).toHaveBeenCalledWith('/api/jobs/job-1/dubbing-texts')
    )

    // First segment source text visible
    expect(screen.getByText(/rapid advancement/)).toBeInTheDocument()

    // Clicking the second card makes it active (editable)
    const card2 = screen.getByTestId('transcript-card-seg-a2')
    await user.click(card2)

    // Textarea now visible for the active segment
    const textarea = await screen.findByLabelText('Dubbed text SPEAKER_01')
    expect(textarea).toBeInTheDocument()

    // Edit the translation
    await user.clear(textarea)
    await user.type(textarea, 'Новы пераклад')

    // Save badge appears
    expect(screen.getByRole('button', { name: /Save/i })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Save/i }))

    await waitFor(() =>
      expect(client.put).toHaveBeenCalledWith(
        '/api/jobs/job-1/dubbing-texts',
        expect.objectContaining({ revision: 'rev-studio-1' }),
      )
    )
  })

  it('plays an audio link and regenerates audio for the active segment', async () => {
    const user = userEvent.setup()
    const client = makeClient()
    render(<HukFlowStudioView client={client} />)

    // Wait for segments
    await screen.findByText('SPEAKER_00')

    // Audio play link present for segment1
    const playLink = screen.getByTitle('Play seg-a1.wav')
    expect(playLink).toHaveAttribute('href', '/api/jobs/job-1/files/aud-1')

    // Activate second card and regenerate
    await user.click(screen.getByTestId('transcript-card-seg-a2'))
    const regenBtn = await screen.findByTitle('Regenerate Audio')
    await user.click(regenBtn)

    await waitFor(() =>
      expect(client.post).toHaveBeenCalledWith(
        '/api/jobs/job-1/dubbing-texts/seg-a2/regenerate',
        expect.objectContaining({ revision: 'rev-studio-1' }),
      )
    )
  })

  it('shows search filter across speaker names and source text', async () => {
    const user = userEvent.setup()
    render(<HukFlowStudioView client={makeClient()} />)

    await screen.findByText('SPEAKER_00')

    const search = screen.getByLabelText('Search transcript')
    await user.type(search, 'paradigm')

    expect(screen.getByText('SPEAKER_00')).toBeInTheDocument()
    expect(screen.queryByText('SPEAKER_01')).not.toBeInTheDocument()

    await user.clear(search)
    expect(screen.getByText('SPEAKER_01')).toBeInTheDocument()
  })

  it('correctly updates subtitle overlay during video playback based on segment time bounds', async () => {
    const client = makeClient()
    const { container } = render(<HukFlowStudioView client={client} />)

    await screen.findByText('SPEAKER_00')

    const videoEl = container.querySelector('video') as HTMLVideoElement
    expect(videoEl).toBeInTheDocument()

    const setTimeAndTrigger = (time: number) => {
      act(() => {
        Object.defineProperty(videoEl, 'currentTime', {
          get: () => time,
          configurable: true,
        })
        fireEvent.timeUpdate(videoEl)
      })
    }

    // Simulate playback at time 2.0 (inside segment1: 1.5s - 4.0s)
    setTimeAndTrigger(2.0)

    const subtitleOverlay = container.querySelector('.subtitle-overlay')
    expect(subtitleOverlay).toBeInTheDocument()
    expect(subtitleOverlay).toHaveTextContent(segment1.translation)

    // Simulate playback at time 10.0 (in gap between segment1 [1.5-4.0] and segment2 [18.0-20.5])
    setTimeAndTrigger(10.0)
    expect(subtitleOverlay).toHaveTextContent('')

    // Simulate playback at time 19.0 (inside segment2: 18.0s - 20.5s)
    setTimeAndTrigger(19.0)
    expect(subtitleOverlay).toHaveTextContent(segment2.translation)
  })

  it('calculates timeline duration dynamically from segment bounds and renders 100% width music track', async () => {
    const client = makeClient({
      get: vi.fn(async (path: string) => {
        if (path === '/api/jobs') return { jobs: [{ id: 'job-1', status: 'succeeded' }] }
        if (path.includes('/dubbing-texts')) return textDocument
        if (path.includes('/files')) return { files: [{ id: 'f1', name: 'custom_background.wav', kind: 'background_audio' }] }
        return {}
      }) as ApiClient['get'],
    })
    const { container } = render(<HukFlowStudioView client={client} />)

    await screen.findByText('SPEAKER_00')

    // Duration should be based on segment max end (20.5s) formatted as 00:00:20:12 instead of hardcoded 60s (00:01:00:00)
    const timecodeDisplay = container.querySelector('.timecode-display')
    expect(timecodeDisplay).toHaveTextContent('00:00:20:12')

    // Music clip should reflect custom_background.wav and have 100% width
    const musicClip = container.querySelector('.music-clip')
    expect(musicClip).toBeInTheDocument()
    expect(musicClip).toHaveTextContent('custom_background.wav')
    expect(musicClip).toHaveStyle({ width: '100%' })
  })
})
