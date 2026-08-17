import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ExportDropdown, triggerFileDownload } from './ExportDropdown'
import type { ApiClient } from '../api/types'

function stubClient(overrides: Partial<ApiClient> = {}): ApiClient {
  return {
    get: vi.fn().mockResolvedValue({}),
    put: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
    upload: vi.fn(),
    subscribeJobEvents: vi.fn(() => () => undefined),
    ...overrides,
  }
}

describe('ExportDropdown component', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      blob: async () => new Blob(['dummy-content'], { type: 'video/mp4' }),
    }))
    // Mock URL.createObjectURL and URL.revokeObjectURL
    window.URL.createObjectURL = vi.fn(() => 'blob:http://localhost/dummy-blob-id')
    window.URL.revokeObjectURL = vi.fn()
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders Export button in disabled state when no project or job is selected', () => {
    render(<ExportDropdown />)
    const exportBtn = screen.getByRole('button', { name: /Export/i })
    expect(exportBtn).toBeInTheDocument()
    expect(exportBtn).toBeDisabled()
    expect(exportBtn).toHaveAttribute('title', 'No project selected for export')
    expect(screen.queryByRole('menu', { name: /Export options/i })).not.toBeInTheDocument()
  })

  it('renders Export button enabled when a project is selected and opens menu on click', async () => {
    const user = userEvent.setup()
    render(<ExportDropdown activeProjectName="Project Alpha" />)

    const exportBtn = screen.getByRole('button', { name: /Export/i })
    expect(exportBtn).toBeEnabled()
    expect(exportBtn).toHaveAttribute('aria-expanded', 'false')

    await user.click(exportBtn)

    expect(exportBtn).toHaveAttribute('aria-expanded', 'true')
    const menu = screen.getByRole('menu', { name: /Export options/i })
    expect(menu).toBeInTheDocument()

    // 2 items: Video and Audio
    expect(screen.getByTestId('export-video-btn')).toBeInTheDocument()
    expect(screen.getByTestId('export-audio-btn')).toBeInTheDocument()
    expect(screen.getByText('Video')).toBeInTheDocument()
    expect(screen.getByText('Audio')).toBeInTheDocument()
    expect(screen.getByText('.MP4')).toBeInTheDocument()
    expect(screen.getByText('.WAV')).toBeInTheDocument()
  })

  it('closes dropdown when clicking outside or pressing Escape', async () => {
    const user = userEvent.setup()
    render(
      <div>
        <div data-testid="outside">Outside</div>
        <ExportDropdown activeProjectName="Project Alpha" />
      </div>,
    )

    const exportBtn = screen.getByRole('button', { name: /Export/i })
    await user.click(exportBtn)
    expect(screen.getByRole('menu')).toBeInTheDocument()

    // Click outside
    await user.click(screen.getByTestId('outside'))
    expect(screen.queryByRole('menu')).not.toBeInTheDocument()

    // Reopen and press Escape
    await user.click(exportBtn)
    expect(screen.getByRole('menu')).toBeInTheDocument()
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('menu')).not.toBeInTheDocument()
  })

  it('downloads video file when Video item is selected', async () => {
    const user = userEvent.setup()
    const files = [
      { id: 'f-vid-1', name: 'Project_Alpha_output.mp4', kind: 'output_video', size: 1000000 },
      { id: 'f-aud-1', name: 'output.wav', kind: 'output_audio', size: 500000 },
    ]

    render(
      <ExportDropdown
        activeJobId="job-100"
        activeProjectName="Project Alpha"
        files={files}
      />,
    )

    await user.click(screen.getByRole('button', { name: /Export/i }))
    await user.click(screen.getByTestId('export-video-btn'))

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/api/jobs/job-100/files/f-vid-1')
    })
    expect(screen.getByText(/Video download started/i)).toBeInTheDocument()
  })

  it('downloads audio file when Audio item is selected', async () => {
    const user = userEvent.setup()
    const files = [
      { id: 'f-vid-1', name: 'Project_Alpha_output.mp4', kind: 'output_video', size: 1000000 },
      { id: 'f-aud-1', name: 'output.wav', kind: 'output_audio', size: 500000 },
    ]

    render(
      <ExportDropdown
        activeJobId="job-100"
        activeProjectName="Project Alpha"
        files={files}
      />,
    )

    await user.click(screen.getByRole('button', { name: /Export/i }))
    await user.click(screen.getByTestId('export-audio-btn'))

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/api/jobs/job-100/files/f-aud-1')
    })
    expect(screen.getByText(/Audio download started/i)).toBeInTheDocument()
  })

  it('fetches files from API if activeJobId is provided without preloaded files', async () => {
    const user = userEvent.setup()
    const client = stubClient({
      get: vi.fn(async (path: string) => {
        if (path === '/api/jobs/job-async-1/files') {
          return {
            files: [
              { id: 'vid-remote-1', name: 'output_video.mp4', kind: 'output_video', size: 2048 },
              { id: 'aud-remote-1', name: 'output.wav', kind: 'output_audio', size: 1024 },
            ],
          }
        }
        return {}
      }) as ApiClient['get'],
    })

    render(
      <ExportDropdown
        client={client}
        activeJobId="job-async-1"
        activeProjectName="Test Job"
      />,
    )

    await user.click(screen.getByRole('button', { name: /Export/i }))
    await user.click(screen.getByTestId('export-video-btn'))

    await waitFor(() => {
      expect(client.get).toHaveBeenCalledWith('/api/jobs/job-async-1/files')
      expect(fetch).toHaveBeenCalledWith('/api/jobs/job-async-1/files/vid-remote-1')
    })
  })

  it('triggerFileDownload fallback triggers direct link click when fetch throws', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Network error')))
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})

    await triggerFileDownload('/fallback.mp4', 'fallback.mp4')
    expect(clickSpy).toHaveBeenCalled()
    clickSpy.mockRestore()
  })
})
