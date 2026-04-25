/**
 * Shared TypeScript interfaces for Orgo API responses and internal types.
 */

export interface OrgoApiResponse {
  [key: string]: unknown;
}

export interface InstanceDetails {
  apiPort?: number;
  publicHost?: string;
  vncHost?: string;
}

export interface ComputerInfo extends OrgoApiResponse {
  id?: string;
  fly_instance_id?: string;
  url?: string;
  instance_details?: InstanceDetails;
  status?: string;
  name?: string;
}

export interface VncPasswordResponse {
  password: string;
}

export interface ScreenshotResponse {
  image: string;
  width?: number;
  height?: number;
}

export interface BashResponse {
  output: string;
}

export interface TerminalMessage {
  type: "input" | "output" | "error" | "exit" | "ping";
  data?: string;
  message?: string;
  code?: number;
}

export type ToolResult = {
  content: Array<
    | { type: "text"; text: string }
    | { type: "image"; data: string; mimeType: string }
  >;
  isError?: boolean;
};
