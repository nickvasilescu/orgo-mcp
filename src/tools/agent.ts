/**
 * AI agent completions tool -- the core agent API.
 *
 * Uses the OpenAI-compatible POST /api/v1/chat/completions endpoint.
 * The AI agent handles the full loop: screenshot -> decide -> act -> repeat.
 */

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getApiKey, resolveComputerId } from "../auth.js";
import { ORGO_V1_BASE } from "../client.js";
import { handleError } from "../errors.js";
import { HttpError } from "../errors.js";

export function registerAgentTools(server: McpServer): void {
  server.tool(
    "orgo_completions",
    "Send an instruction to an AI agent controlling an Orgo computer. The agent sees the screen, clicks, types, and runs commands autonomously until the task is done. Returns the agent's final response and a thread_id for follow-up.",
    {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      instruction: z.string().min(1).describe("What the AI agent should do (natural language)"),
      model: z.string().default("claude-sonnet-4.6").describe("Model: claude-sonnet-4.6 or claude-opus-4.6"),
      thread_id: z.string().optional().describe("Thread ID to continue a previous conversation"),
      max_steps: z.number().int().min(1).max(500).optional().describe("Max agent steps (default: 100)"),
      anthropic_key: z.string().optional().describe("Your Anthropic API key for BYOK mode (no Orgo credits used)"),
    },
    async ({ computer_id, instruction, model, thread_id, max_steps, anthropic_key }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);

        const body: Record<string, unknown> = {
          model,
          messages: [{ role: "user", content: instruction }],
          computer_id: id,
        };
        if (thread_id) body.thread_id = thread_id;
        if (max_steps !== undefined) body.max_steps = max_steps;

        const headers: Record<string, string> = {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        };
        if (anthropic_key) headers["X-Anthropic-Key"] = anthropic_key;

        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 300000);

        try {
          const response = await fetch(`${ORGO_V1_BASE}/chat/completions`, {
            method: "POST",
            headers,
            body: JSON.stringify(body),
            signal: controller.signal,
          });

          if (!response.ok) {
            let detail = "";
            try {
              const errJson = (await response.json()) as Record<string, unknown>;
              detail = (errJson.error as string) || "";
            } catch {
              // ignore
            }
            throw new HttpError(response.status, detail, response.statusText);
          }

          const data = await response.json();
          return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
        } finally {
          clearTimeout(timer);
        }
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );
}
