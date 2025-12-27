# ==============================================================================
# Orgo MCP Server Dockerfile
# Multi-stage build for minimal, secure production image
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and build the package
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (better layer caching)
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the package with dependencies
COPY orgo_mcp.py ./
RUN pip install --no-cache-dir .

# ------------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Labels for container metadata
LABEL org.opencontainers.image.title="Orgo MCP Server" \
      org.opencontainers.image.description="MCP server for AI agents to control virtual computers via Orgo" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Nick Vasilescu" \
      org.opencontainers.image.source="https://github.com/nickvasilescu/orgo-mcp" \
      org.opencontainers.image.licenses="MIT"

# Security: Run as non-root user
RUN groupadd --gid 1000 mcp && \
    useradd --uid 1000 --gid mcp --shell /bin/false --create-home mcp

# Runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # MCP server configuration
    MCP_TRANSPORT=http \
    PORT=8000

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
WORKDIR /app
COPY --chown=mcp:mcp orgo_mcp.py ./

# Switch to non-root user
USER mcp

# Expose the default port
EXPOSE 8000

# Health check - verify the server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('localhost', ${PORT:-8000})); s.close()" || exit 1

# Default command - run the MCP server
CMD ["python", "orgo_mcp.py"]
