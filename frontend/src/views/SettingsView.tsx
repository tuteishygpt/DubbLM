import { FormEvent, useEffect, useState } from 'react'
import type { ApiClient, ConfigResponse, JsonValue, OptionsResponse, SelectOption } from '../api/types'
import { SchemaField } from '../components/SchemaField'

export function SettingsView({ client, config }: { client: ApiClient; config: ConfigResponse }) {
  const fields = config.schema.fields.filter((field) => !field.workflow && field.scope !== 'workflow')
  const [values, setValues] = useState<Record<string, JsonValue>>(() => ({ ...config.values }))
  const [options, setOptions] = useState<OptionsResponse>({})
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  useEffect(() => { client.get<OptionsResponse>('/api/options').then(setOptions).catch((reason) => setError(String(reason))) }, [client])

  async function save(event: FormEvent) {
    event.preventDefault(); setError('')
    try {
      const normalized = { ...values }
      for (const field of fields) if (['list', 'object', 'structured'].includes(field.type) && typeof normalized[field.name] === 'string') normalized[field.name] = JSON.parse(normalized[field.name] as string) as JsonValue
      await client.put('/api/config', { revision: config.revision, values: normalized }); setMessage('Settings saved')
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }

  return <section><h1>Settings</h1>{error && <p role="alert">{error}</p>}{message && <p role="status">{message}</p>}<form onSubmit={save}>
    {fields.map((field) => <SchemaField key={field.name} field={field} value={values[field.name]} options={normalizeOptions(field.options ?? options[field.options_key ?? field.name])} onChange={(value) => setValues((current) => ({ ...current, [field.name]: value }))} />)}
    <button type="submit">Save settings</button>
  </form></section>
}

function normalizeOptions(value: unknown): SelectOption[] {
  return Array.isArray(value) ? value.map((item) => typeof item === 'string' ? { value: item, label: item } : item as SelectOption) : []
}
