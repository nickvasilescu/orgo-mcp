// HTTP entry point. `handler.fetch(request)` is a web-standard Request to
// Response, so it drops into a Next.js route, a Hono route, or node:http
// unchanged. Each request builds its own server with that caller's key, so a
// shared deployment holds nothing between callers.

import { createMcpHandler, type McpHttpHandler } from '@modelcontextprotocol/server'
import { createServer, resolveBaseUrl, VERSION } from './server.js'
import { loadSpec, type Spec } from './spec.js'
import { applyDefaultComputer, applyPolicy, buildTools, type Tool } from './tools.js'

export interface HttpOptions {
  /** Where the caller's Orgo key lives. Defaults to the `Authorization` bearer token. */
  apiKey?: (request: Request) => string | undefined
  /**
   * Pins a computer for this caller so action tools can omit `id`. Defaults to
   * the `X-Orgo-Default-Computer-Id` header, then a `computer_id` query param,
   * for clients that only let you configure a URL.
   */
  defaultComputerId?: (request: Request) => string | undefined
  /**
   * Origin to call. Defaults to `ORGO_API_BASE_URL`, then the spec's own server
   * (`https://www.orgo.ai`). Set it to reach another environment or a domain
   * alias. A function receives the request, so a host serving this route from
   * inside the API can call its own origin without configuration.
   */
  baseUrl?: string | ((request: Request) => string | undefined)
  /** Reported to clients. Defaults to this package's version. */
  version?: string
  timeoutMs?: number
}

interface Ready {
  spec: Spec
  tools: Tool[]
}

const bearer = (request: Request) => request.headers.get('authorization')?.replace(/^Bearer /i, '')

const pinnedComputer = (request: Request) =>
  request.headers.get('x-orgo-default-computer-id') ??
  new URL(request.url).searchParams.get('computer_id') ??
  undefined

async function prepare(): Promise<Ready> {
  const { spec } = await loadSpec()
  // Every HTTP caller can pin a computer, so the shared tool list always marks
  // `id` optional. The value itself arrives per request.
  return { spec, tools: applyDefaultComputer(applyPolicy(buildTools(spec)), true) }
}

export function createHttpHandler(options: HttpOptions = {}): McpHttpHandler {
  let ready: Promise<Ready> | undefined

  // Loaded on the first request and shared after. A failure is not cached, or a
  // single bad load would poison the process until it restarts.
  const load = () =>
    (ready ??= prepare().catch((error) => {
      ready = undefined
      throw error
    }))

  return createMcpHandler(async (ctx) => {
    const { spec, tools } = await load()
    const request = ctx.requestInfo
    const apiKey = (request && (options.apiKey ?? bearer)(request)) ?? ''
    const defaultComputerId = request && (options.defaultComputerId ?? pinnedComputer)(request)
    const origin = typeof options.baseUrl === 'function' ? request && options.baseUrl(request) : options.baseUrl

    return createServer(
      {
        apiKey,
        baseUrl: resolveBaseUrl(spec, origin ?? process.env.ORGO_API_BASE_URL),
        timeoutMs: options.timeoutMs ?? 120_000,
        defaultComputerId,
      },
      tools,
      { version: options.version ?? VERSION, instructions: spec.info.description },
    )
  })
}
