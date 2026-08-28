// Turns OpenAPI operations into MCP tools.
//
// One operation, one tool. Path params, query params and JSON body fields are
// flattened into a single argument list, because a model calling
// `orgo_resize_computer({ id, mem_gb })` reads better than one nesting a body.
// Every name, description, type and enum below comes from the spec.

import { deref, type Operation, type Schema, type Spec } from './spec.js'

export interface Arg {
  /** `whole-body` means the argument is the request body, not a field in it. */
  in: 'path' | 'query' | 'body' | 'whole-body'
  /** Multipart file field: the argument is a local path to read. */
  binary?: boolean
}

export interface Tool {
  name: string
  title?: string
  description: string
  tags: string[]
  method: string
  path: string
  /** 'none' when the operation takes no request body. */
  body: 'json' | 'multipart' | 'none'
  args: Record<string, Arg>
  inputSchema: Schema
  /** The success response the spec promises, used to drop everything it doesn't. */
  response?: Schema
  annotations: { readOnlyHint: boolean; destructiveHint: boolean; idempotentHint: boolean; openWorldHint: true }
}

const METHODS = ['get', 'post', 'put', 'patch', 'delete']

export function buildTools(spec: Spec): Tool[] {
  return Object.entries(spec.paths).flatMap(([path, item]) =>
    Object.entries(item)
      // An operation with no operationId has no name to derive; skipping one
      // beats throwing and taking all 43 tools down with it.
      .filter(([method, op]) => METHODS.includes(method) && op.operationId)
      .map(([method, op]) => toTool(spec, path, method, op as Operation & { operationId: string })),
  )
}

function toTool(spec: Spec, path: string, method: string, op: Operation & { operationId: string }): Tool {
  const args: Record<string, Arg> = {}
  const properties: Record<string, Schema> = {}
  const required: string[] = []

  const add = (name: string, schema: Schema, arg: Arg, isRequired: boolean) => {
    args[name] = arg
    properties[name] = schema
    if (isRequired) required.push(name)
  }

  for (const param of op.parameters ?? []) {
    const schema = deref(param.schema ?? { type: 'string' }, spec)
    add(
      param.name,
      describe(schema, param.description),
      { in: param.in },
      param.in === 'path' || !!param.required,
    )
  }

  const { body, schema: bodySchema } = pickBody(spec, op)
  if (bodySchema?.properties) {
    const bodyRequired = new Set(bodySchema.required ?? [])
    for (const [name, raw] of Object.entries(bodySchema.properties)) {
      const binary = body === 'multipart' && raw.format === 'binary'
      const schema = binary
        ? { type: 'string', description: `Absolute path to a local file. ${raw.description ?? ''}`.trim() }
        : raw
      add(name, schema, { in: 'body', ...(binary ? { binary: true } : {}) }, bodyRequired.has(name))
    }
  } else if (bodySchema) {
    // A body that isn't an object of named fields (an array, a bare string)
    // passes through whole rather than being dropped on the floor.
    add('body', bodySchema, { in: 'whole-body' }, !!op.requestBody?.required)
  }

  const readOnly = method === 'get'
  return {
    name: toolName(op.operationId),
    title: op.summary,
    description: [op.description ?? op.summary, bodySchema?.description].filter(Boolean).join('\n\n'),
    tags: op.tags ?? [],
    method: method.toUpperCase(),
    path,
    body,
    args,
    inputSchema: {
      type: 'object',
      properties,
      required,
      // Closing a body the spec declares open would reject documents the API accepts.
      additionalProperties: bodySchema?.additionalProperties === true,
    },
    response: successSchema(spec, op),
    annotations: {
      readOnlyHint: readOnly,
      // MCP defaults destructiveHint to true; claiming false on a write is a
      // promise of additive-only behaviour that bash and exec cannot keep.
      destructiveHint: !readOnly,
      idempotentHint: readOnly || method === 'put' || method === 'delete',
      openWorldHint: true,
    },
  }
}

/** JSON wins when an operation accepts several body encodings: it is the one we can describe field by field. */
function pickBody(spec: Spec, op: Operation): { body: Tool['body']; schema?: Schema } {
  const content = op.requestBody?.content
  if (!content) return { body: 'none' }
  if (content['application/json']) return { body: 'json', schema: deref(content['application/json'].schema, spec) }
  if (content['multipart/form-data']) return { body: 'multipart', schema: deref(content['multipart/form-data'].schema, spec) }
  return { body: 'none' }
}

function successSchema(spec: Spec, op: Operation): Schema | undefined {
  const ok = op.responses?.['200'] ?? op.responses?.['201']
  const schema = ok?.content?.['application/json']?.schema
  return schema && deref(schema, spec)
}

function describe(schema: Schema, description?: string): Schema {
  return description && !schema.description ? { ...schema, description } : schema
}

/** `createComputer` -> `orgo_create_computer`. Mechanical, so a new operationId needs no decision from anyone. */
export function toolName(operationId: string): string {
  return `orgo_${operationId.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase()}`
}

/**
 * A pinned computer lets an agent say "screenshot the desktop" without repeating
 * the same id on every call. Only arguments that name a computer are eligible,
 * so a workspace or file id is never mistaken for one. Stdio takes the value
 * from ORGO_DEFAULT_COMPUTER_ID; an HTTP host reads it per request.
 */
const defaultTarget = (tool: Tool) =>
  tool.path.startsWith('/computers/{id}') ? 'id' : 'desktopId' in tool.args ? 'desktopId' : null

/**
 * The argument stops being required, or the SDK rejects the call before we can
 * fill it. This shapes the shared tool list, so it is a deployment-wide switch:
 * the per-request value arrives later, through Config.
 */
export function applyDefaultComputer(tools: Tool[], enabled = !!process.env.ORGO_DEFAULT_COMPUTER_ID): Tool[] {
  if (!enabled) return tools

  return tools.map((tool) => {
    const target = defaultTarget(tool)
    if (!target || !tool.inputSchema.required?.includes(target)) return tool
    const required = tool.inputSchema.required.filter((name) => name !== target)
    return { ...tool, inputSchema: { ...tool.inputSchema, required } }
  })
}

export function withDefaults(
  tool: Tool,
  input: Record<string, unknown>,
  fallback?: string,
): Record<string, unknown> {
  const target = fallback ? defaultTarget(tool) : null
  return target && input[target] === undefined ? { ...input, [target]: fallback } : input
}

/** ORGO_TAGS / ORGO_READ_ONLY narrow the surface without touching client config. */
export function applyPolicy(tools: Tool[]): Tool[] {
  // A safety switch has to fail safe: anything but an explicit off means on, so
  // ORGO_READ_ONLY=1 cannot quietly leave every DELETE tool registered.
  const readOnly = !/^(|false|0|no|off)$/i.test((process.env.ORGO_READ_ONLY ?? '').trim())
  const wanted = (process.env.ORGO_TAGS ?? '')
    .split(',')
    .map((tag) => tag.trim().toLowerCase())
    .filter(Boolean)

  return tools.filter(
    (tool) =>
      (!readOnly || tool.annotations.readOnlyHint) &&
      (!wanted.length || tool.tags.some((tag) => wanted.includes(tag.toLowerCase()))),
  )
}
