/**
 * Screen action tools -- mouse, keyboard, and screenshot control.
 */

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getApiKey, resolveComputerId } from "../auth.js";
import { computerAction } from "../client.js";
import { handleError } from "../errors.js";
import type { ScreenshotResponse } from "../types.js";
import { registerOrgoTool } from "./registry.js";

// Model coordinate space -- matches orgo-web's CUA
const MODEL_WIDTH = 1280;
const MODEL_HEIGHT = 720;

// Cache actual VM resolutions: computer_id -> [width, height]
const vmResolutionCache = new Map<string, [number, number]>();

function scaleCoords(computerId: string, x: number, y: number): [number, number] {
  const vm = vmResolutionCache.get(computerId);
  if (!vm || (vm[0] === MODEL_WIDTH && vm[1] === MODEL_HEIGHT)) {
    return [x, y];
  }
  return [
    Math.round(x * (vm[0] / MODEL_WIDTH)),
    Math.round(y * (vm[1] / MODEL_HEIGHT)),
  ];
}

export function registerActionTools(server: McpServer): void {
  registerOrgoTool(server, {
    name: "orgo_screenshot",
    title: "Take Screenshot",
    description: "Take a screenshot of the full VM display (all windows, desktop). Returns an image.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
    },
    toolsets: ["screen"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ computer_id }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = (await computerAction("GET", id, "screenshot", apiKey, {
          direct: true,
        })) as unknown as ScreenshotResponse;

        // Cache VM resolution for coordinate scaling
        if (data.width && data.height) {
          vmResolutionCache.set(id, [data.width, data.height]);
        }

        // Clean base64 image data
        let imageB64 = (data.image || "")
          .trim()
          .replace(/\n/g, "")
          .replace(/\r/g, "")
          .replace(/ /g, "");

        // Fix padding
        const pad = imageB64.length % 4;
        if (pad) imageB64 += "=".repeat(4 - pad);

        return {
          content: [
            {
              type: "image" as const,
              data: imageB64,
              mimeType: "image/png",
            },
          ],
        };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_click",
    title: "Click",
    description: "Click at pixel (x, y) coordinates on the VM display. Coordinates are in 1280x720 model space.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      x: z.number().int().min(0).describe("X coordinate (pixels from left)"),
      y: z.number().int().min(0).describe("Y coordinate (pixels from top)"),
      button: z.enum(["left", "right"]).default("left").describe("Mouse button"),
      double: z.boolean().default(false).describe("Double-click if true"),
    },
    toolsets: ["screen"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ computer_id, x, y, button, double: dbl }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const [realX, realY] = scaleCoords(id, x, y);
        const payload: Record<string, unknown> = { x: realX, y: realY, button };
        if (dbl) payload.double = true;
        await computerAction("POST", id, "click", apiKey, { json: payload });
        const action = dbl ? "Double-clicked" : button === "right" ? "Right-clicked" : "Clicked";
        return { content: [{ type: "text" as const, text: `${action} at (${x}, ${y})` }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_type",
    title: "Type Text",
    description: "Type text at the current cursor position on the VM.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      text: z.string().min(1).describe("Text to type at cursor position"),
    },
    toolsets: ["screen"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ computer_id, text }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        await computerAction("POST", id, "type", apiKey, { json: { text } });
        const preview = text.length > 50 ? text.substring(0, 50) + "..." : text;
        return { content: [{ type: "text" as const, text: `Typed: ${preview}` }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_key",
    title: "Press Key",
    description: "Press a key or combo: Enter, Tab, Escape, ctrl+c, alt+Tab, ctrl+shift+s, F1-F12.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      key: z.string().min(1).describe("Key or combo: Enter, Tab, Escape, ctrl+c, alt+Tab"),
    },
    toolsets: ["screen"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ computer_id, key }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        await computerAction("POST", id, "key", apiKey, { json: { key } });
        return { content: [{ type: "text" as const, text: `Pressed: ${key}` }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_scroll",
    title: "Scroll",
    description: "Scroll the VM display up or down.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      direction: z.enum(["up", "down"]).describe("Scroll direction"),
      amount: z.number().int().min(1).max(20).default(3).describe("Scroll clicks (1-20)"),
    },
    toolsets: ["screen"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ computer_id, direction, amount }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        await computerAction("POST", id, "scroll", apiKey, { json: { direction, amount } });
        return { content: [{ type: "text" as const, text: `Scrolled ${direction} by ${amount}` }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_drag",
    title: "Drag",
    description: "Drag from (start_x, start_y) to (end_x, end_y). Coordinates in 1280x720 model space.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      start_x: z.number().int().min(0).describe("Start X coordinate"),
      start_y: z.number().int().min(0).describe("Start Y coordinate"),
      end_x: z.number().int().min(0).describe("End X coordinate"),
      end_y: z.number().int().min(0).describe("End Y coordinate"),
      duration: z.number().min(0.1).max(5.0).default(0.5).describe("Drag duration in seconds"),
    },
    toolsets: ["screen"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ computer_id, start_x, start_y, end_x, end_y, duration }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const [realSx, realSy] = scaleCoords(id, start_x, start_y);
        const [realEx, realEy] = scaleCoords(id, end_x, end_y);
        await computerAction("POST", id, "drag", apiKey, {
          json: {
            start_x: realSx,
            start_y: realSy,
            end_x: realEx,
            end_y: realEy,
            button: "left",
            duration,
          },
        });
        return {
          content: [{ type: "text" as const, text: `Dragged (${start_x},${start_y}) -> (${end_x},${end_y})` }],
        };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });
}
