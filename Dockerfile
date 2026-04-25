# ==============================================================================
# Orgo MCP Server Dockerfile (Node.js)
# Multi-stage build for minimal, secure production image
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and compile TypeScript
# ------------------------------------------------------------------------------
FROM node:22-slim AS builder

WORKDIR /build

# Copy dependency files first (better layer caching)
COPY package.json package-lock.json* tsconfig.json ./

# Install all dependencies (including dev for build)
RUN npm ci

# Copy source and build
COPY src/ ./src/
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# ------------------------------------------------------------------------------
FROM node:22-slim AS runtime

# Labels for container metadata
LABEL org.opencontainers.image.title="Orgo MCP Server" \
      org.opencontainers.image.description="MCP server for AI agents to control virtual computers via Orgo" \
      org.opencontainers.image.version="3.0.0" \
      org.opencontainers.image.vendor="Nick Vasilescu" \
      org.opencontainers.image.source="https://github.com/nickvasilescu/orgo-mcp" \
      org.opencontainers.image.licenses="MIT"

# Security: Run as non-root user
RUN groupadd --gid 1000 mcp && \
    useradd --uid 1000 --gid mcp --shell /bin/false --create-home mcp

WORKDIR /app

# Copy package files and install production deps only
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev && npm cache clean --force

# Copy compiled JavaScript from builder
COPY --from=builder /build/dist/ ./dist/

# Runtime environment
ENV MCP_TRANSPORT=http \
    PORT=8000 \
    NODE_ENV=production

# Switch to non-root user
USER mcp

# Expose the default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "const http=require('http');const r=http.get('http://localhost:'+(process.env.PORT||8000)+'/health',s=>{process.exit(s.statusCode===200?0:1)});r.on('error',()=>process.exit(1))"

# Default command
CMD ["node", "dist/index.js"]
