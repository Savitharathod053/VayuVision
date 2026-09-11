FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

# Install dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy application files
COPY backend/ ./backend/
COPY utils/ ./utils/
COPY models/ ./models/
COPY data/ ./data/
COPY frontend/dist/ ./frontend/dist/

EXPOSE 10000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
