// Turns a tool call into an HTTP request against the Orgo API.

import { readFile } from 'node:fs/promises'
import { basename } from 'node:path'
import type { Tool } from './tools.js'

export interface Config {
  apiKey: string
  baseUrl: string
  timeoutMs: number
  /** Fills `id` on computer tools when the caller omits it. */
  defaultComputerId?: string
}

export interface Reply {
  status: number
  contentType: string
  json?: unknown
  text: string
}

/** A build-log stream never ends on its own; return what arrived and move on. */
const STREAM_BUDGET_MS = 10_000
const STREAM_LIMIT_BYTES = 64 * 1024

const RETRIES = Number(process.env.ORGO_RETRIES ?? 2)
const BACKOFF_MS = 250

export async function call(
  tool: Tool,
  input: Record<string, unknown>,
  config: Config,
  signal: AbortSignal,
): Promise<Reply> {
  const url = new URL(config.baseUrl + fillPath(tool, input))
  const headers: Record<string, string> = { Authorization: `Bearer ${config.apiKey}` }

  for (const [name, arg] of Object.entries(tool.args))
    if (arg.in === 'query' && input[name] !== undefined) url.searchParams.set(name, String(input[name]))

  const body = await buildBody(tool, input, headers)

  const res = await send(
    () =>
      fetch(url, {
        method: tool.method,
        headers,
        body,
        signal: AbortSignal.any([signal, AbortSignal.timeout(config.timeoutMs)]),
      }),
    // Only the spec's idempotent methods are safe to repeat. Retrying a click or
    // a bash command could run it twice, so those get one shot.
    tool.annotations.idempotentHint,
    signal,
  )

  const contentType = res.headers.get('content-type') ?? ''
  const text = contentType.includes('text/event-stream') ? await readStream(res) : await res.text()

  let json: unknown
  if (contentType.includes('json')) {
    try {
      json = JSON.parse(text)
    } catch {
      // Fall through to the raw text; a malformed body is still worth showing.
    }
  }

  return { status: res.status, contentType, json, text }
}

/** Rides out a proxy restart or a dropped connection, doubling the wait each time. */
async function send(attempt: () => Promise<Response>, retryable: boolean, signal: AbortSignal): Promise<Response> {
  for (let tries = 0; ; tries++) {
    const last = !retryable || tries >= RETRIES || signal.aborted
    try {
      const res = await attempt()
      if (res.status < 500 || last) return res
    } catch (error) {
      if (last) throw error
    }
    await new Promise((resume) => setTimeout(resume, BACKOFF_MS * 2 ** tries))
  }
}

function fillPath(tool: Tool, input: Record<string, unknown>): string {
  return tool.path.replace(/\{([^}]+)\}/g, (_, name: string) => {
    const value = input[name]
    if (value === undefined) throw new Error(`Missing required path parameter: ${name}`)
    return encodeURIComponent(String(value))
  })
}

async function buildBody(
  tool: Tool,
  input: Record<string, unknown>,
  headers: Record<string, string>,
): Promise<string | FormData | undefined> {
  if (tool.body === 'none') return undefined

  const whole = Object.keys(tool.args).find((name) => tool.args[name]?.in === 'whole-body')
  if (whole) {
    headers['Content-Type'] = 'application/json'
    return input[whole] === undefined ? undefined : JSON.stringify(input[whole])
  }

  const fields = Object.entries(tool.args).filter(([name, arg]) => arg.in === 'body' && input[name] !== undefined)

  if (tool.body === 'multipart') {
    const form = new FormData()
    for (const [name, arg] of fields) {
      if (arg.binary) {
        const path = String(input[name])
        form.append(name, new Blob([await readFile(path)]), basename(path))
      } else {
        form.append(name, String(input[name]))
      }
    }
    return form // fetch sets the multipart boundary itself.
  }

  // An open body schema carries keys the spec never enumerated, so anything the
  // caller passed that isn't a known argument goes through untouched.
  const extra = tool.inputSchema.additionalProperties
    ? Object.entries(input).filter(([name]) => !Object.hasOwn(tool.args, name))
    : []

  headers['Content-Type'] = 'application/json'
  return JSON.stringify(Object.fromEntries([...extra, ...fields.map(([name]) => [name, input[name]])]))
}

async function readStream(res: Response): Promise<string> {
  if (!res.body) return ''
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  // One timer for the whole stream. Checking the clock between reads would never
  // fire, because a quiet stream parks inside the awaited read.
  const stop = () => reader.cancel().catch(() => {})
  const budget = setTimeout(stop, STREAM_BUDGET_MS).unref()
  let out = ''

  try {
    while (out.length < STREAM_LIMIT_BYTES) {
      const { done, value } = await reader.read()
      if (done) break
      out += decoder.decode(value, { stream: true })
    }
  } catch {
    // Timed out or aborted mid-stream; the partial log is still the answer.
  } finally {
    clearTimeout(budget)
    await stop()
  }

  return out
}
