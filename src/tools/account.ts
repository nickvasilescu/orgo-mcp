/**
 * Account tools -- user profile, credits, and transactions.
 */

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getApiKey } from "../auth.js";
import { apiRequest } from "../client.js";
import { handleError } from "../errors.js";

export function registerAccountTools(server: McpServer): void {
  server.tool(
    "orgo_get_profile",
    "Get the current user's profile: ID, email, name, avatar, and subscription tier.",
    {},
    async () => {
      try {
        const apiKey = getApiKey();
        const profile = await apiRequest("GET", "user/profile", apiKey) as Record<string, unknown>;
        const subscription = await apiRequest("GET", "user/subscription", apiKey) as Record<string, unknown>;
        profile.tier = subscription.tier || "free";
        return { content: [{ type: "text" as const, text: JSON.stringify(profile, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_get_credits",
    "Get the current credit balance and subscription tier. Balance is in cents (e.g. 5000 = $50.00).",
    {},
    async () => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", "credits", apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_get_transactions",
    "Get credit transaction history: purchases, usage, and balance changes.",
    {},
    async () => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", "credits/transactions", apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );
}
