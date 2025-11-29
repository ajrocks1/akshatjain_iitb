FROM python:3.11-slim

# Set env vars for logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
# poppler-utils: for PDF -> Image conversion
# libgl1: for OpenCV (just in case)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY . /app

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
