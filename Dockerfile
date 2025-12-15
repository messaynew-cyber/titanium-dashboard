# Build Frontend
FROM node:18-alpine as frontend-builder
WORKDIR /frontend_build
COPY frontend/package*.json ./
COPY frontend/tsconfig*.json ./
COPY frontend/vite.config.ts ./
COPY frontend/tailwind.config.js ./
COPY frontend/index.html ./
RUN npm ci
COPY frontend/src ./src
RUN npm run build

# Runtime
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt backend_reqs.txt
COPY algo/requirements.txt algo_reqs.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r backend_reqs.txt && pip install --no-cache-dir -r algo_reqs.txt
COPY backend /app/backend
COPY algo /app/algo
COPY --from=frontend-builder /frontend_build/dist /app/static
RUN mkdir -p /app/TITANIUM_V1_FIXED/state && mkdir -p /app/TITANIUM_V1_FIXED/logs
ENV PYTHONPATH=/app
ENV PORT=80
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "80"]
