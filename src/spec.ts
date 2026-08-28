// The Orgo OpenAPI document is the single source of truth for every tool this
// server exposes. Nothing about the API is hand-written here.
//
// Freshness order: fresh cache -> network -> stale cache -> bundled snapshot.
// The snapshot means the server still starts on a plane or in a locked-down CI
// box; the network fetch means a new endpoint in the docs is a new tool on the
// next restart, with no release.

import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

export interface Schema {
  type?: string
  description?: string
  properties?: Record<string, Schema>
  required?: string[]
  format?: string
  [key: string]: unknown
}

export interface Operation {
  operationId?: string
  summary?: string
  description?: string
  tags?: string[]
  parameters?: { name: string; in: 'path' | 'query'; required?: boolean; description?: string; schema?: Schema }[]
  requestBody?: { required?: boolean; content: Record<string, { schema: Schema }> }
  responses?: Record<string, { content?: Record<string, { schema: Schema }> }>
}

export interface Spec {
  info: { title: string; version: string; description?: string }
  servers?: { url: string }[]
  paths: Record<string, Record<string, Operation>>
  components?: Record<string, Record<string, unknown>>
}

const here = dirname(fileURLToPath(import.meta.url))
export const SNAPSHOT = join(here, '..', 'openapi.snapshot.json')
const CACHE = join(process.env.XDG_CACHE_HOME || join(homedir(), '.cache'), 'orgo-mcp', 'openapi.json')

export const SPEC_URL = process.env.ORGO_SPEC_URL || 'https://docs.orgo.ai/api-reference/openapi.json'
export const SPEC_MAX_AGE_MS = Number(process.env.ORGO_SPEC_MAX_AGE_MS ?? 24 * 60 * 60 * 1000)

/** Loads the spec and reports where it came from, for the startup log. */
export async function loadSpec(): Promise<{ spec: Spec; source: string }> {
  const cached = await readCache()

  if (cached && cached.ageMs < SPEC_MAX_AGE_MS) return { spec: cached.spec, source: 'cache' }

  const live = await fetchSpec()
  if (live) {
    await writeCache(live)
    return { spec: live, source: SPEC_URL }
  }

  if (cached) return { spec: cached.spec, source: 'cache (stale, network unreachable)' }
  return { spec: JSON.parse(await readFile(SNAPSHOT, 'utf8')), source: 'bundled snapshot' }
}

export async function fetchSpec(url = SPEC_URL): Promise<Spec | null> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(5000) })
    if (!res.ok) return null
    const spec = (await res.json()) as Spec
    // A login page or an error envelope can still arrive as 200 JSON. Reject it
    // here or it poisons the cache and every start fails until the cache ages out.
    return spec?.paths && typeof spec.paths === 'object' && spec.info?.version ? spec : null
  } catch {
    return null
  }
}

/**
 * Inlines local `$ref`s so tool schemas are self-contained. MCP clients get a
 * plain JSON Schema, not one that points at `#/components`. A ref already open
 * on the current branch collapses to a bare object, which keeps recursive
 * schemas (templates reference themselves) from expanding forever.
 */
export function deref<T>(node: T, spec: Spec, open: ReadonlySet<string> = new Set()): T {
  if (Array.isArray(node)) return node.map((n) => deref(n, spec, open)) as T
  if (!node || typeof node !== 'object') return node

  const obj = node as Record<string, unknown>
  const ref = typeof obj.$ref === 'string' ? obj.$ref : null

  if (ref) {
    if (open.has(ref)) return { type: 'object' } as T
    const target = resolve(ref, spec)
    if (!target) return { type: 'object' } as T
    const { $ref, ...siblings } = obj
    return { ...deref(target, spec, new Set([...open, ref])), ...siblings } as T
  }

  return Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, deref(v, spec, open)])) as T
}

function resolve(ref: string, spec: Spec): Record<string, unknown> | null {
  if (!ref.startsWith('#/')) return null
  let node: unknown = spec
  for (const part of ref.slice(2).split('/')) {
    if (!node || typeof node !== 'object') return null
    node = (node as Record<string, unknown>)[part.replace(/~1/g, '/').replace(/~0/g, '~')]
  }
  return node && typeof node === 'object' ? (node as Record<string, unknown>) : null
}

async function readCache() {
  try {
    const [raw, info] = await Promise.all([readFile(CACHE, 'utf8'), stat(CACHE)])
    return { spec: JSON.parse(raw) as Spec, ageMs: Date.now() - info.mtimeMs }
  } catch {
    return null
  }
}

async function writeCache(spec: Spec) {
  try {
    await mkdir(dirname(CACHE), { recursive: true })
    await writeFile(CACHE, JSON.stringify(spec))
  } catch {
    // A read-only home directory costs us the cache, not the server.
  }
}
