// Hosted entry for the Render deployment at orgo-mcp.onrender.com. Serves the
// same spec-generated tools as the stdio binary over Streamable HTTP, each
// request carrying its caller's own key, so the deployment holds nothing
// between callers. Not part of the npm package; `orgo-mcp-server` never runs this.

import { createServer } from 'node:http'
import { toNodeHandler } from '@modelcontextprotocol/node'
import { createHttpHandler } from './build/server.js'

const handler = createHttpHandler({
  // This deployment has always documented X-Orgo-API-Key; the library default
  // is the Authorization bearer. Accept both so no existing config breaks.
  apiKey: (request) =>
    request.headers.get('x-orgo-api-key') ??
    request.headers.get('authorization')?.replace(/^Bearer /i, '') ??
    undefined,
})

const mcp = toNodeHandler(handler)
const port = Number(process.env.PORT ?? 10000)

createServer((req, res) => {
  const path = new URL(req.url ?? '/', 'http://x').pathname
  if (path === '/health') {
    res.writeHead(200, { 'content-type': 'text/plain' }).end('ok')
  } else if (path === '/' && req.method === 'GET') {
    res
      .writeHead(200, { 'content-type': 'text/plain' })
      .end('Orgo MCP. Connect an MCP client to POST /mcp with your API key. https://docs.orgo.ai/guides/mcp\n')
  } else {
    mcp(req, res)
  }
}).listen(port, () => console.error(`[orgo-mcp] listening on :${port}, MCP at /mcp`))
