export declare function sanitizeForMcp(value: unknown, seen?: WeakSet<object>): unknown;
export declare function jsonText(value: unknown): string;
export declare function compactProjection(value: unknown, seen?: WeakSet<object>): unknown;
export declare function jsonTextCompact(value: unknown): string;
export declare function applyLimit(data: Record<string, unknown>, arrayKey: string, limit: number | undefined): Record<string, unknown>;
