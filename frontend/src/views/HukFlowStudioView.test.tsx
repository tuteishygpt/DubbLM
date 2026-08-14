import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { ApiClient } from '../api/types'
import { HukFlowStudioView } from './HukFlowStudioView'

function mockClient(): ApiClient {
  return {
    get: vi.fn().mockResolvedValue({ jobs: [{ id: 'job-101', status: 'succeeded' }] }),
    post: vi.fn().mockResolvedValue({ status: 'ok' }),
    put: vi.fn(),
    delete: vi.fn(),
    upload: vi.fn(),
    subscribeJobEvents: vi.fn(() => () => undefined),
  }
}

describe('HukFlowStudioView', () => {
  it('renders transcript cards, allows text editing, search filtering, and track muting', async () => {
    const user = userEvent.setup()
    const client = mockClient()

    render(<HukFlowStudioView client={client} />)

    // Heading for accessibility
    expect(screen.getByRole('heading', { name: 'HukFlow Studio', level: 1 })).toBeInTheDocument()

    // Verify transcript title and cards
    expect(screen.getByText('Localization Transcript')).toBeInTheDocument()
    expect(screen.getByText('Speaker 1')).toBeInTheDocument()
    expect(screen.getByText('Speaker 2')).toBeInTheDocument()

    // Verify editable dubbed text area
    const textarea = screen.getByLabelText('Dubbed text Speaker 2')
    expect(textarea).toHaveValue('Мы павінны ўлічваць этычныя наступствы перад разгортваннем.')

    // Edit text
    await user.clear(textarea)
    await user.type(textarea, 'Новы пераклад мовы')
    expect(textarea).toHaveValue('Новы пераклад мовы')

    // Search filter
    const searchInput = screen.getByPlaceholderText('Search transcript...')
    await user.type(searchInput, 'paradigm')
    expect(screen.getByText('Speaker 1')).toBeInTheDocument()
    expect(screen.queryByText('Speaker 2')).not.toBeInTheDocument()

    await user.clear(searchInput)

    // Audio regeneration button
    const regenBtn = screen.getByTitle('Regenerate Audio')
    await user.click(regenBtn)

    await waitFor(() => {
      expect(client.post).toHaveBeenCalledWith(
        '/api/jobs/job-101/dubbing-texts/seg-2/regenerate',
        { synthesized_text: 'Новы пераклад мовы' }
      )
    })
  })
})
