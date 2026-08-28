#!/usr/bin/env node
// orgo-mcp — every Orgo API operation, as MCP tools, straight from the spec.

import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { StdioServerTransport } from '@modelcontextprotocol/server/stdio'
import { createServer, resolveBaseUrl } from './server.js'
import { loadSpec } from './spec.js'
import { applyDefaultComputer, applyPolicy, buildTools } from './tools.js'

// stdout carries JSON-RPC; anything we want a human to read goes to stderr.
const log = (message: string) => console.error(`[orgo-mcp] ${message}`)

/**
 * Same key, same precedence, same profile names as the Orgo CLI, so `orgo login`
 * is enough and no `sk_live_` key has to sit in an MCP client's config file.
 */
async function resolveAuth(): Promise<{ apiKey?: string; apiBaseUrl?: string }> {
  if (process.env.ORGO_API_KEY) return { apiKey: process.env.ORGO_API_KEY }
  try {
    const file = JSON.parse(await readFile(join(homedir(), '.orgo', 'credentials.json'), 'utf8'))
    return file.profiles?.[process.env.ORGO_PROFILE || file.current || 'default'] ?? {}
  } catch {
    return {}
  }
}

async function main() {
  const { apiKey, apiBaseUrl } = await resolveAuth()
  if (!apiKey) {
    log('No API key. Set ORGO_API_KEY, or run `orgo login`. Keys: https://www.orgo.ai/workspaces')
    process.exit(1)
  }

  const { spec, source } = await loadSpec()
  const tools = applyDefaultComputer(applyPolicy(buildTools(spec)))
  const baseUrl = resolveBaseUrl(spec, process.env.ORGO_API_BASE_URL || apiBaseUrl)

  const server = createServer(
    {
      apiKey,
      baseUrl,
      timeoutMs: Number(process.env.ORGO_TIMEOUT_MS ?? 120_000),
      defaultComputerId: process.env.ORGO_DEFAULT_COMPUTER_ID,
    },
    tools,
    { instructions: spec.info.description },
  )

  await server.connect(new StdioServerTransport())
  log(`${tools.length} tools from Orgo API ${spec.info.version} (${source}) → ${baseUrl}`)
}

main().catch((error) => {
  log(String(error instanceof Error ? error.stack ?? error.message : error))
  process.exit(1)
})
