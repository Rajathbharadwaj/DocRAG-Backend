# Use Python 3.11 slim image as base
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the source code
COPY src/ /app/src/
COPY requirements.txt .
# Use uv to install Python dependencies
RUN uv pip install --system -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"] 