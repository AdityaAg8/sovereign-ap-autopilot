# Placeholder image for apx/parse's PaddleOCR 3.x service (PRD §10 M0.4): a minimal
# FastAPI healthcheck stub standing in until M3.1 builds the real GPU-backed container.
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.118.0 uvicorn==0.37.0
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
