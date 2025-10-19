# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent files
COPY movie_specialist_agent.py .
COPY ticket_master_agent.py .
COPY vendor_agent.py .
COPY orchestrator_agent.py .
COPY main.py .
COPY .env.example .

# Copy data directory (only JSON files)
COPY data/vendor.json data/

# Set environment variable for Python output
ENV PYTHONUNBUFFERED=1

# Run the main application
CMD ["python", "main.py"]
