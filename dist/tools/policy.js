export const ALL_TOOLSETS = [
    "core",
    "admin",
    "screen",
    "shell",
    "files",
];
function parseCsvSet(value) {
    if (!value)
        return undefined;
    const entries = value
        .split(",")
        .map((entry) => entry.trim())
        .filter(Boolean);
    return entries.length > 0 ? new Set(entries) : undefined;
}
function envFlag(name) {
    const value = process.env[name];
    if (!value)
        return false;
    return ["1", "true", "yes", "on"].includes(value.toLowerCase());
}
export function isReadOnlyMode() {
    return envFlag("ORGO_READ_ONLY");
}
export function isToolEnabled(tool) {
    const enabledTools = parseCsvSet(process.env.ORGO_ENABLED_TOOLS);
    if (enabledTools && !enabledTools.has(tool.name))
        return false;
    const disabledTools = parseCsvSet(process.env.ORGO_DISABLED_TOOLS);
    if (disabledTools?.has(tool.name))
        return false;
    const enabledToolsets = parseCsvSet(process.env.ORGO_TOOLSETS);
    if (enabledToolsets && !tool.toolsets.some((toolset) => enabledToolsets.has(toolset))) {
        return false;
    }
    if (isReadOnlyMode() && tool.annotations.readOnlyHint !== true) {
        return false;
    }
    return true;
}
export function getToolPolicySummary() {
    const toolsets = process.env.ORGO_TOOLSETS || ALL_TOOLSETS.join(",");
    const enabledTools = process.env.ORGO_ENABLED_TOOLS || "(all registered tools)";
    const disabledTools = process.env.ORGO_DISABLED_TOOLS || "(none)";
    const readOnly = isReadOnlyMode() ? "enabled" : "disabled";
    return [
        `Read-only mode: ${readOnly}`,
        `Enabled toolsets: ${toolsets}`,
        `Enabled tool allowlist: ${enabledTools}`,
        `Disabled tools: ${disabledTools}`,
    ].join("\n");
}
//# sourceMappingURL=policy.js.map