import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ApiRequestError } from '../api/client'
import type { ApiClient, ConfigResponse } from '../api/types'
import { SettingsView } from './SettingsView'

const config: ConfigResponse = { revision: 'rev-4', values: { model: 'm1', retries: 2, enabled: true, speakers: ['A'], metadata: { owner: 'team' } }, schema: { fields: [
  { name: 'model', label: 'Model', type: 'select', options_key: 'models' },
  { name: 'retries', label: 'Retries', type: 'number' },
  { name: 'enabled', label: 'Enabled', type: 'boolean' },
  { name: 'speakers', label: 'Speakers', type: 'list' },
  { name: 'metadata', label: 'Metadata', type: 'object' },
] } }

function api(conflict = false): ApiClient {
  return {
    get: vi.fn().mockResolvedValue({ models: [{ value: 'm1', label: 'Model One' }, { value: 'm2', label: 'Model Two' }] }),
    put: conflict ? vi.fn().mockRejectedValue(new ApiRequestError('Revision conflict', 'revision_conflict', 409)) : vi.fn().mockResolvedValue({ ...config, revision: 'rev-5' }),
    post: vi.fn(), delete: vi.fn(), upload: vi.fn(), subscribeJobEvents: vi.fn(() => () => undefined),
  }
}

describe('SettingsView', () => {
  it('renders every schema field and saves structured/list values with the revision', async () => {
    const user = userEvent.setup(); const client = api()
    render(<SettingsView client={client} config={config} />)
    await screen.findByRole('option', { name: 'Model Two' })
    for (const label of ['Model', 'Retries', 'Enabled', 'Speakers', 'Metadata']) expect(screen.getByLabelText(label)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Speakers'), { target: { value: '["A","B"]' } })
    fireEvent.change(screen.getByLabelText('Metadata'), { target: { value: '{"owner":"ops"}' } })
    await user.selectOptions(screen.getByLabelText('Model'), 'm2')
    await user.click(screen.getByRole('button', { name: 'Save settings' }))
    await waitFor(() => expect(client.put).toHaveBeenCalledWith('/api/config', { revision: 'rev-4', values: expect.objectContaining({ model: 'm2', speakers: ['A', 'B'], metadata: { owner: 'ops' } }) }))
  })

  it('surfaces revision conflicts', async () => {
    const user = userEvent.setup(); render(<SettingsView client={api(true)} config={config} />)
    await screen.findByRole('option', { name: 'Model Two' }); await user.click(screen.getByRole('button', { name: 'Save settings' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Revision conflict')
  })
})
