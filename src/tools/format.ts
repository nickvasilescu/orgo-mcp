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

export function jsonText(value: unknown): string {
  return JSON.stringify(sanitizeForMcp(value), null, 2);
}
