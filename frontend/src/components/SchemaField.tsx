import type { JsonValue, SchemaFieldDefinition, SelectOption } from '../api/types'

export function SchemaField({ field, value, options = [], onChange }: { field: SchemaFieldDefinition; value: JsonValue; options?: SelectOption[]; onChange(value: JsonValue): void }) {
  const label = field.label ?? field.name
  if (field.type === 'boolean') return <label><input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />{label}</label>
  if (field.type === 'select') return <label>{label}<select value={String(value ?? '')} onChange={(event) => onChange(event.target.value)}>{options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
  if (field.type === 'list' || field.type === 'object' || field.type === 'structured') return <label>{label}<textarea value={typeof value === 'string' ? value : JSON.stringify(value)} onChange={(event) => onChange(event.target.value)} /></label>
  return <label>{label}<input type={field.type === 'number' ? 'number' : 'text'} min={field.minimum} max={field.maximum} value={String(value ?? '')} onChange={(event) => onChange(field.type === 'number' ? Number(event.target.value) : event.target.value)} /></label>
}
