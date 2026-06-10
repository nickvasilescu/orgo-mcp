/**
 * Unified error handling for Orgo MCP tools.
 */
export interface FetchErrorLike {
    status?: number;
    statusText?: string;
    body?: string;
}
/**
 * Format errors into user-friendly messages with actionable guidance.
 */
export declare function handleError(e: unknown): string;
/**
 * Custom HTTP error class for API responses.
 */
export declare class HttpError extends Error {
    status: number;
    detail: string;
    statusText: string;
    constructor(status: number, detail: string, statusText?: string);
}
