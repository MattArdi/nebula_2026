# ---- Stage 1: build the frontend ----
FROM node:20-slim AS frontend-build
WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
# The deployed backend serves this build itself (see the runtime stage's
# StaticFiles mount), so API calls are same-origin -- an empty base means
# apiClient.js's `${API_BASE}${path}` collapses to a plain relative URL.
ENV VITE_API_BASE_URL=""
RUN npm run build

# ---- Stage 2: the backend, serving the built frontend ----
FROM python:3.11-slim AS runtime
WORKDIR /app

# catboost and xgboost's compiled wheels link against libgomp at runtime;
# python:3.11-slim doesn't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/api/requirements.txt backend/api/requirements.txt
RUN pip install --no-cache-dir -r backend/api/requirements.txt

# The four subsystem pipelines (rules/features/model code + shipped
# artifacts/weights) plus the API layer that wraps them via subprocess.
COPY backend/ backend/

# Built once in stage 1; served by main.py's StaticFiles mount at "/".
COPY --from=frontend-build /app/dashboard/dist/ static/

# Cloud Run injects PORT at runtime (defaults to 8080 locally too).
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT}"]
