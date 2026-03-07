# ============================================================================
# Document Intelligence Refinery — Docker Image
# ============================================================================
# Multi-stage build for a minimal production image.
#
# Build:
#   docker build -t refinery .
#
# Run (with .env):
#   docker run --env-file .env -v $(pwd)/data:/refinery/data refinery run data/sample.pdf
#
# Interactive shell:
#   docker run --env-file .env -it refinery bash
# ============================================================================

FROM python:3.11-slim AS base

# System dependencies for pdfplumber/PyMuPDF/chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /refinery

# Install Python dependencies first (layer cache)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || pip install --no-cache-dir .

# Copy source code
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || pip install --no-cache-dir -e .

# Initialise the database
RUN python -m src.cli init-db || true

# Default: show help
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
