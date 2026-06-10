import { isToolEnabled } from "./policy.js";
export function registerOrgoTool(server, tool) {
    if (!isToolEnabled(tool))
        return;
    server.registerTool(tool.name, {
        title: tool.title,
        description: tool.description,
        inputSchema: tool.inputSchema,
        annotations: {
            ...tool.annotations,
            title: tool.annotations.title || tool.title,
        },
        _meta: {
            "orgo/toolsets": tool.toolsets,
        },
    }, tool.handler);
}
//# sourceMappingURL=registry.js.map