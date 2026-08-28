// Public entry point. `createHttpHandler` is what a host wants; the rest is
// here for anything wiring its own transport.

import { McpServer, fromJsonSchema } from '@modelcontextprotocol/server'
import { call, type Config } from './client.js'
import { toResult } from './result.js'
import { SPEC_MAX_AGE_MS, type Spec } from './spec.js'
import { withDefaults, type Tool } from './tools.js'

export { createHttpHandler, type HttpOptions } from './http.js'
export { loadSpec, fetchSpec, type Spec, type Schema } from './spec.js'
export { buildTools, applyPolicy, applyDefaultComputer, type Tool } from './tools.js'
export type { Config } from './client.js'

/** Kept in step with package.json by a test, so no bundler has to resolve a path at runtime. */
export const VERSION = '1.1.1'

export interface ServerOptions {
  /** Reported to clients. Defaults to this package's version. */
  version?: string
  /** Orientation for the model, normally the spec's own `info.description`. */
  instructions?: string
}

export function createServer(config: Config, tools: Tool[], options: ServerOptions = {}): McpServer {
  const server = new McpServer(
    {
      name: 'orgo',
      title: 'Orgo',
      version: options.version ?? VERSION,
      icons: [{ src: 'https://www.orgo.ai/icon.svg', mimeType: 'image/svg+xml', sizes: ['any'] }],
    },
    {
      instructions: options.instructions,
      // The tool list is the same for every caller and fixed for the life of the
      // process, so it is worth a shared cache for as long as the spec is fresh.
      cacheHints: {
        'tools/list': { ttlMs: SPEC_MAX_AGE_MS, cacheScope: 'public' },
        'server/discover': { ttlMs: SPEC_MAX_AGE_MS, cacheScope: 'public' },
      },
    },
  )

  for (const tool of tools) {
    server.registerTool(
      tool.name,
      {
        title: tool.title,
        description: tool.description,
        inputSchema: fromJsonSchema<Record<string, unknown>>(tool.inputSchema),
        annotations: tool.annotations,
      },
      async (input, ctx) => {
        try {
          const args = withDefaults(tool, input ?? {}, config.defaultComputerId)
          const reply = await call(tool, args, config, ctx.mcpReq.signal)
          return await toResult(reply, config.baseUrl, tool.response)
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error)
          return { content: [{ type: 'text' as const, text: `Request failed: ${message}` }], isError: true }
        }
      },
    )
  }

  return server
}

/** The API base the spec points at, with an optional origin override. */
export function resolveBaseUrl(spec: Spec, origin?: string): string {
  const base = origin ? `${origin}/api` : (spec.servers?.[0]?.url ?? 'https://www.orgo.ai/api')
  return base.replace(/\/$/, '')
}
