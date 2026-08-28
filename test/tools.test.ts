// Everything here runs against the bundled snapshot. No network, no API key.

import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import type { AddressInfo } from 'node:net'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'
import { shape, toResult } from '../src/result.js'
import { VERSION } from '../src/server.js'
import { deref, SNAPSHOT, type Spec } from '../src/spec.js'
import { call } from '../src/client.js'
import { applyDefaultComputer, applyPolicy, buildTools, withDefaults } from '../src/tools.js'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const spec = JSON.parse(await readFile(SNAPSHOT, 'utf8')) as Spec
const tools = buildTools(spec)
const byName = new Map(tools.map((t) => [t.name, t]))

test('every operation becomes exactly one uniquely named tool', () => {
  const operations = Object.values(spec.paths).flatMap((item) =>
    Object.keys(item).filter((m) => ['get', 'post', 'put', 'patch', 'delete'].includes(m)),
  )
  assert.equal(tools.length, operations.length)
  assert.equal(byName.size, tools.length)
})

test('tool names are snake_case and within the 64-character limit', () => {
  for (const tool of tools) assert.match(tool.name, /^orgo_[a-z0-9_]{1,58}$/)
})

test('path, query and body params flatten into one argument list', () => {
  const resize = byName.get('orgo_resize_computer')
  assert.ok(resize)
  assert.deepEqual(resize.args.id, { in: 'path' })
  assert.deepEqual(resize.args.mem_gb, { in: 'body' })
  assert.deepEqual(resize.inputSchema.required, ['id'])
  assert.equal(resize.inputSchema.properties?.mem_gb?.type, 'integer')

  const publish = byName.get('orgo_publish_template')
  assert.ok(publish)
  assert.deepEqual(publish.args.auto_build, { in: 'query' })
})

// The flat namespace only works while no two parameters of one operation share a
// name. Nothing in the code prevents that, so the spec is checked here instead.
test('no operation has a name collision across path, query and body', () => {
  for (const [path, item] of Object.entries(spec.paths)) {
    for (const [method, op] of Object.entries(item)) {
      if (!op.operationId) continue
      const names = [
        ...(op.parameters ?? []).map((p) => p.name),
        ...Object.keys(op.requestBody?.content?.['application/json']?.schema?.properties ?? {}),
      ]
      assert.equal(new Set(names).size, names.length, `${method.toUpperCase()} ${path} has a duplicate parameter name`)
    }
  }
})

test('schemas are fully dereferenced', () => {
  const create = byName.get('orgo_create_computer')
  assert.ok(create)
  assert.deepEqual(create.inputSchema.required, ['workspace_id', 'name'])
  assert.deepEqual(create.inputSchema.properties?.ram?.enum, [4, 8, 16, 32, 64])
  assert.ok(!JSON.stringify(tools).includes('$ref'))
})

test('a body the spec declares open stays open', () => {
  assert.equal(byName.get('orgo_publish_template')?.inputSchema.additionalProperties, true)
  assert.equal(byName.get('orgo_create_computer')?.inputSchema.additionalProperties, false)
})

test('multipart file fields ask for a local path', () => {
  const upload = byName.get('orgo_upload_file')
  assert.ok(upload)
  assert.equal(upload.body, 'multipart')
  assert.equal(upload.args.file?.binary, true)
  assert.match(String(upload.inputSchema.properties?.file?.description), /local file/)
})

test('annotations follow the HTTP method', () => {
  assert.equal(byName.get('orgo_get_computer')?.annotations.readOnlyHint, true)
  assert.equal(byName.get('orgo_get_computer')?.annotations.destructiveHint, false)
  assert.equal(byName.get('orgo_delete_computer')?.annotations.destructiveHint, true)
  assert.equal(byName.get('orgo_execute_bash')?.annotations.destructiveHint, true)
})

test('title and description come from the spec, separately', () => {
  const create = byName.get('orgo_create_computer')
  assert.equal(create?.title, 'Create computer')
  assert.match(String(create?.description), /^Creates a new virtual computer/)
})

test('an operation with no operationId is skipped, not fatal', () => {
  const patched: Spec = {
    ...spec,
    paths: { ...spec.paths, '/nameless': { get: { summary: 'no id' } } },
  }
  assert.equal(buildTools(patched).length, tools.length)
})

test('ORGO_READ_ONLY fails safe and ORGO_TAGS narrows by tag', (t) => {
  t.after(() => {
    delete process.env.ORGO_READ_ONLY
    delete process.env.ORGO_TAGS
  })

  process.env.ORGO_TAGS = 'workspaces'
  assert.deepEqual(
    applyPolicy(tools).map((tool) => tool.name),
    ['orgo_list_workspaces', 'orgo_create_workspace', 'orgo_get_workspace', 'orgo_delete_workspace'],
  )

  const readTools = ['orgo_list_workspaces', 'orgo_get_workspace']
  for (const on of ['true', '1', 'yes', 'TRUE']) {
    process.env.ORGO_READ_ONLY = on
    assert.deepEqual(applyPolicy(tools).map((tool) => tool.name), readTools, `${on} should enable read-only`)
  }
  for (const off of ['false', '0', 'no', 'off', '']) {
    process.env.ORGO_READ_ONLY = off
    assert.equal(applyPolicy(tools).length, 4, `${off} should leave read-only off`)
  }
})

test('deref collapses cycles instead of expanding forever', () => {
  const cyclic: Spec = {
    info: { title: 't', version: '1' },
    paths: {},
    components: { schemas: { Node: { type: 'object', properties: { child: { $ref: '#/components/schemas/Node' } } } } },
  }
  const resolved = deref({ $ref: '#/components/schemas/Node' }, cyclic) as Record<string, any>
  assert.deepEqual(resolved.properties.child, { type: 'object' })
})

// The live API returns `image` as an origin-relative path, which is the case a
// naive "is it a URL?" check misses.
test('a screenshot link is fetched and returned as an image block', async () => {
  const png = Buffer.from('89504e470d0a1a0a', 'hex')
  const server = createServer((_req, res) => {
    res.writeHead(200, { 'content-type': 'image/png' })
    res.end(png)
  })
  await new Promise<void>((done) => server.listen(0, '127.0.0.1', done))
  const origin = `http://127.0.0.1:${(server.address() as AddressInfo).port}/api`
  const reply = (image: string) => ({ status: 200, contentType: 'application/json', json: { image }, text: '' })

  try {
    for (const link of ['/api/storage/shot.png', `${origin}/storage/shot.png`]) {
      const [content] = (await toResult(reply(link), origin)).content
      assert.equal(content?.type, 'image', `${link} should become an image`)
      assert.equal((content as { mimeType: string }).mimeType, 'image/png')
      assert.equal((content as { data: string }).data, png.toString('base64'))
    }

    const [unreachable] = (await toResult(reply('/api/storage/gone.png'), 'http://127.0.0.1:1')).content
    assert.equal(unreachable?.type, 'text', 'an unreachable link falls back to text')
  } finally {
    server.close()
  }
})

test('a validation failure keeps its per-field errors', async () => {
  const text = JSON.stringify({ error: 'Template validation failed', errors: [{ field: 'hardware.cpu', hint: 'use 1, 2, 4' }] })
  const [content] = (await toResult({ status: 422, contentType: 'application/json', json: JSON.parse(text), text }, 'x')).content
  assert.match(String((content as { text: string }).text), /hardware\.cpu/)
})

test('a pinned computer relaxes the schema and fills the value', () => {
  const relaxed = new Map(applyDefaultComputer(tools, true).map((tool) => [tool.name, tool]))
  const shot = relaxed.get('orgo_get_screenshot')!
  assert.deepEqual(shot.inputSchema.required, [], 'id is no longer required, or the SDK rejects the call first')
  assert.deepEqual(withDefaults(shot, {}, 'vm-1'), { id: 'vm-1' })
  assert.deepEqual(withDefaults(shot, { id: 'explicit' }, 'vm-1'), { id: 'explicit' }, 'an explicit id always wins')
  assert.deepEqual(withDefaults(shot, {}, undefined), {}, 'no pin, no fill')

  // A workspace id is not a computer id.
  const ws = relaxed.get('orgo_get_workspace')!
  assert.deepEqual(ws.inputSchema.required, ['id'])
  assert.deepEqual(withDefaults(ws, {}, 'vm-1'), {})

  // Two callers on one deployment must not see each other's pin.
  assert.deepEqual(withDefaults(shot, {}, 'vm-a'), { id: 'vm-a' })
  assert.deepEqual(withDefaults(shot, {}, 'vm-b'), { id: 'vm-b' })

  // Off by default, so stdio without the env var is unchanged.
  assert.deepEqual(applyDefaultComputer(tools, false), tools)
})

test('idempotent requests retry through a 503; writes do not', async () => {
  let getHits = 0
  let postHits = 0
  const server = createServer((req, res) => {
    const first = req.method === 'GET' ? getHits++ === 0 : postHits++ === 0
    res.writeHead(first ? 503 : 200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ ok: !first }))
  })
  await new Promise<void>((done) => server.listen(0, '127.0.0.1', done))
  const config = { apiKey: 'k', baseUrl: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, timeoutMs: 5000 }
  const signal = new AbortController().signal

  try {
    const read = byName.get('orgo_get_computer')!
    assert.equal((await call(read, { id: 'x' }, config, signal)).status, 200, 'a GET rides out one 503')
    assert.equal(getHits, 2)

    const write = byName.get('orgo_execute_bash')!
    assert.equal((await call(write, { id: 'x', command: 'ls' }, config, signal)).status, 503, 'bash must never run twice')
    assert.equal(postHits, 1)
  } finally {
    server.close()
  }
})

// The README quotes counts that no build step regenerates, so they are pinned here.
test('README numbers match the code', async () => {
  const readme = await readFile(join(root, 'README.md'), 'utf8')

  for (const [, file, claimed] of readme.matchAll(/\| `(src\/\w+\.ts)` \| (\d+) \|/g)) {
    const actual = (await readFile(join(root, file!), 'utf8')).split('\n').length - 1
    assert.equal(Number(claimed), actual, `README claims ${claimed} lines for ${file}, actual ${actual}`)
  }

  for (const [, claimed] of readme.matchAll(/(\d+) tools/g))
    assert.equal(Number(claimed), tools.length, `README says "${claimed} tools", actual ${tools.length}`)
})

test('shaping drops undocumented fields but never empties a mismatched payload', () => {
  const computer = { type: 'object', properties: { id: { type: 'string' }, name: { type: 'string' } } }
  const wrapper = { type: 'object', properties: { workspaces: { type: 'array', items: computer } } }

  assert.deepEqual(shape({ id: '1', name: 'a', vnc_password_encrypted: 'x', host_id: 'h' }, computer), { id: '1', name: 'a' })
  assert.deepEqual(shape({ workspaces: [{ id: '1', name: 'a', host_id: 'h' }] }, wrapper), { workspaces: [{ id: '1', name: 'a' }] })

  // The live API wraps this list in `projects`, which the spec calls `workspaces`.
  const mismatched = { projects: [{ id: '1', host_id: 'h' }] }
  assert.deepEqual(shape(mismatched, wrapper), mismatched, 'a schema matching nothing must not empty the payload')
  assert.deepEqual(shape(mismatched, undefined), mismatched)
})

test('VERSION matches package.json', async () => {
  const pkg = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
  assert.equal(VERSION, pkg.version)
})

test('README follows house style', async () => {
  const readme = await readFile(join(root, 'README.md'), 'utf8')
  assert.ok(!readme.includes('—'), 'use a spaced hyphen, not an em dash')
  const hype = readme.match(/\b(simply|blazing|seamless|effortless|supercharge)\b/i)
  assert.equal(hype, null, `remove "${hype?.[0]}"`)
})

// orgo-web returns the same array under both `workspaces` (canonical) and
// `projects` (pre-rename alias). The spec documents only the canonical key, so
// shaping drops the duplicate instead of sending it to the model twice.
test('the deprecated projects alias is dropped, not doubled', async () => {
  const list = byName.get('orgo_list_workspaces')!
  const row = { id: 'w1', name: 'prod', status: 'active', desktops: [], host_id: 'h9' }
  const dual = { workspaces: [row], projects: [row] }

  const [content] = (await toResult(
    { status: 200, contentType: 'application/json', json: dual, text: JSON.stringify(dual) },
    'https://www.orgo.ai/api',
    list.response,
  )).content
  const shaped = JSON.parse((content as { text: string }).text)

  assert.deepEqual(Object.keys(shaped), ['workspaces'], 'only the canonical key survives')
  assert.equal(shaped.workspaces[0].name, 'prod')
  assert.equal(shaped.workspaces[0].host_id, undefined, 'undocumented fields still stripped')

  // Before orgo-web ships the alias, the response has only `projects`, which the
  // schema describes none of. That must pass through rather than empty out.
  const legacy = { projects: [row] }
  const [before] = (await toResult(
    { status: 200, contentType: 'application/json', json: legacy, text: JSON.stringify(legacy) },
    'https://www.orgo.ai/api',
    list.response,
  )).content
  assert.deepEqual(JSON.parse((before as { text: string }).text), legacy, 'legacy shape survives untouched')
})
