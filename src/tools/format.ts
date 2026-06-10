const SENSITIVE_KEY_RE = /password|secret|token|api[_-]?key|apikey|private[_-]?key|credential/i;
const SENSITIVE_VALUE_RE = /\bsk_live_[A-Za-z0-9_-]+/g;

export function sanitizeForMcp(value: unknown, seen = new WeakSet<object>()): unknown {
  if (typeof value === "string") {
    return value.replace(SENSITIVE_VALUE_RE, "sk_live_[redacted]");
  }

  if (value === null || typeof value !== "object") {
    return value;
  }

  if (seen.has(value)) {
    return "[Circular]";
  }
  seen.add(value);

  if (Array.isArray(value)) {
    return value.map((entry) => sanitizeForMcp(entry, seen));
  }

  const sanitized: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) {
    sanitized[key] = SENSITIVE_KEY_RE.test(key) ? "[redacted]" : sanitizeForMcp(entry, seen);
  }
  return sanitized;
}

// No pretty-print indentation: MCP responses are consumed by models, and the
// 2-space indent was pure token overhead on every response.
export function jsonText(value: unknown): string {
  return JSON.stringify(sanitizeForMcp(value));
}

// Keys preserved by compactProjection — identity, status, timestamps, key
// resource fields, top-level containers, and standard response wrappers.
// Drops everything else (instance_details, template_build_id, vnc_password_encrypted,
// user_id, etc.) to cut typical response sizes 5-10× for agent contexts.
const COMPACT_KEEP_KEYS = new Set([
  // identity
  "id",
  "name",
  "status",
  "url",
  // timestamps
  "created_at",
  "updated_at",
  // computer essentials
  "cpu",
  "ram",
  "os",
  "disk_size_gb",
  "fly_instance_id",
  "resolution",
  // file essentials
  "size",
  "path",
  "filename",
  "type",
  "mime_type",
  // top-level containers (preserve structure)
  "projects",
  "workspaces",
  "computers",
  "desktops",
  "files",
  // meta / aggregates
  "count",
  "total",
  "truncated",
  "meta",
  // standard response envelopes
  "success",
  "error",
  "error_type",
  "message",
  "output",
]);

export function compactProjection(value: unknown, seen = new WeakSet<object>()): unknown {
  if (value === null || typeof value !== "object") {
    return value;
  }

  if (seen.has(value)) {
    return "[Circular]";
  }
  seen.add(value);

  if (Array.isArray(value)) {
    return value.map((entry) => compactProjection(entry, seen));
  }

  const compact: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (COMPACT_KEEP_KEYS.has(key)) {
      compact[key] = compactProjection(entry, seen);
    }
  }
  return compact;
}

export function jsonTextCompact(value: unknown): string {
  return jsonText(compactProjection(value));
}

// Client-side limit for list endpoints. Orgo's API doesn't paginate server-side,
// so callers that pass `limit` get a slice plus `total` and `truncated` metadata.
// When `limit` is undefined, the data is returned unchanged (backward compatible).
export function applyLimit(
  data: Record<string, unknown>,
  arrayKey: string,
  limit: number | undefined
): Record<string, unknown> {
  if (limit === undefined) return data;
  const arr = data[arrayKey];
  if (!Array.isArray(arr)) return data;
  if (arr.length <= limit) return { ...data, count: arr.length, total: arr.length, truncated: false };
  return {
    ...data,
    [arrayKey]: arr.slice(0, limit),
    count: limit,
    total: arr.length,
    truncated: true,
  };
}
