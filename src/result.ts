// Shapes an HTTP reply into MCP tool content.
//
// Two things earn special handling: errors, so the model sees the API's own
// message instead of a wall of JSON, and screenshots, so it sees the screen
// instead of a link.

import type { CallToolResult } from '@modelcontextprotocol/server'
import type { Reply } from './client.js'
import type { Schema } from './spec.js'

type Content = CallToolResult['content'][number]

const DATA_URL = /^data:(image\/[\w+.-]+);base64,(.+)$/s
const RAW = !!process.env.ORGO_RAW

export async function toResult(reply: Reply, origin: string, schema?: Schema): Promise<CallToolResult> {
  if (reply.status >= 400) {
    // A validation failure carries per-field detail in `errors[]`, which is the
    // half a model needs to fix its own call. Keep the whole body when it's there.
    const body = reply.json as { error?: string; errors?: unknown[] } | undefined
    const message = body?.errors?.length ? reply.text : (body?.error ?? reply.text.slice(0, 500))
    return { content: [{ type: 'text', text: `HTTP ${reply.status}: ${message || reply.contentType}` }], isError: true }
  }

  const image = reply.json && typeof reply.json === 'object' ? (reply.json as { image?: unknown }).image : undefined
  if (typeof image === 'string') {
    const content = await toImage(image, origin)
    if (content) return { content: [content] }
  }

  if (reply.json === undefined) return { content: [{ type: 'text', text: reply.text }] }
  return { content: [{ type: 'text', text: JSON.stringify(shape(reply.json, schema)) }] }
}

/**
 * Keeps only what the spec says a response contains. The API also returns
 * internal state the docs never promise (encrypted VNC passwords, host and
 * server ids, boot details), which is both noise and 80% of the bytes.
 *
 * A schema that describes none of the keys in hand is a schema describing
 * something else, so the value passes through rather than being emptied. Spec
 * drift costs verbosity here, never data. Set ORGO_RAW=1 to skip shaping.
 */
export function shape(value: unknown, schema?: Schema): unknown {
  if (!schema || RAW) return value
  if (Array.isArray(value)) return schema.items ? value.map((item) => shape(item, schema.items as Schema)) : value
  if (!value || typeof value !== 'object') return value

  const declared = properties(schema)
  if (!declared) return value

  const entries = Object.entries(value as Record<string, unknown>)
  const kept = entries.filter(([key]) => key in declared)
  if (!kept.length) return value

  return Object.fromEntries(kept.map(([key, nested]) => [key, shape(nested, declared[key])]))
}

/** `allOf` composition is how the spec says "a Workspace, plus its computers". */
function properties(schema: Schema): Record<string, Schema> | null {
  if (schema.properties) return schema.properties
  const members = schema.allOf as Schema[] | undefined
  return members ? Object.assign({}, ...members.map((member) => properties(member) ?? {})) : null
}

/** A screenshot arrives as a data URL, a link to the stored capture (usually origin-relative), or bare base64. */
async function toImage(value: string, origin: string): Promise<Content | null> {
  const [, mime, base64] = value.match(DATA_URL) ?? []
  if (mime && base64) return { type: 'image', mimeType: mime, data: base64 }

  if (value.startsWith('/') || /^https?:\/\//.test(value)) {
    try {
      // No credentials: the stored capture is served by an unauthenticated proxy,
      // and the URL comes from a response body rather than from us.
      const res = await fetch(new URL(value, origin), { signal: AbortSignal.timeout(15_000) })
      const mimeType = res.headers.get('content-type') ?? ''
      if (!res.ok || !mimeType.startsWith('image/')) return null
      return { type: 'image', mimeType, data: Buffer.from(await res.arrayBuffer()).toString('base64') }
    } catch {
      return null // Fall back to returning the link as text.
    }
  }

  return /^[A-Za-z0-9+/=\s]+$/.test(value) && value.length > 256
    ? { type: 'image', mimeType: 'image/png', data: value.replace(/\s/g, '') }
    : null
}
