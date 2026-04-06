# FastAPI LLM Async API

Production-style FastAPI starter for async text generation using **Gemini** (cloud) or **Ollama/gemma3** (local).

## What this repo includes

- **Dual LLM support:** Gemini API or local Ollama (gemma3)
- Async FastAPI endpoint (`/v1/generate`)
- Pydantic request/response validation
- Shared async `httpx` client with connection pooling
- `asyncio.Semaphore` to cap concurrent outbound LLM calls
- Request ID middleware for tracing
- Docker + Docker Compose
- Optional Nginx config for multi-instance load balancing
- GitHub Actions CI ready

## Architecture

### Single instance (local or cloud)
```text
Users -> FastAPI -> Ollama (local) or Gemini API
```

### Multi-instance (production)
```text
Users -> Nginx (load balancer) -> FastAPI 1 / FastAPI 2 / ... -> LLM backend
```

## Why this design

This repo assumes the heavy LLM work happens on an external service (Ollama, Gemini), not your FastAPI server. The FastAPI app focuses on:

- validation & request shaping
- outbound async API calls
- concurrency control (semaphore)
- clean JSON responses
- request tracing

## Quick start

### 1. Clone and configure

```bash
git clone <repo>
cd Text-2-Text
cp .env.example .env
```

### 2a. Use Ollama (local, free)

Ensure **Ollama is running locally** with gemma3 model:

```bash
# Download and run Ollama (if not already running)
ollama pull gemma3
ollama serve  # Listens on http://localhost:11434
```

Then in `.env`:
```env
# Leave GEMINI_API_KEY empty or unset
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=gemma3
```

### 2b. Use Gemini (cloud, requires API key)

In `.env`:
```env
GEMINI_API_KEY=your_real_api_key_here
GEMINI_DEFAULT_MODEL=gemini-2.5-flash
# Leave OLLAMA_BASE_URL empty or unset
```

### 3. Run locally

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\venv\Scripts\Activate
pip install -r requirements.txt
# Use --workers 1 on Windows to avoid socket errors
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 4. Call the API

**Minimal request (uses default model):**
```bash
curl -X POST "http://127.0.0.1:8000/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize FastAPI in one paragraph.",
    "temperature": 0.2,
    "max_output_tokens": 200
  }'
```

**Full request (explicit model + system instruction):**
```bash
curl -X POST "http://127.0.0.1:8000/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Translate to technical documentation: Use an async HTTP client.",
    "system_instruction": "Be concise. Use bullet points.",
    "model": "gemma3",
    "temperature": 0.2,
    "max_output_tokens": 300,
    "top_p": 0.95,
    "top_k": 40,
    "response_mime_type": "text/plain"
  }'
```

**PowerShell (Windows):**
```powershell
$payload = @{
    prompt = "Hello, what is FastAPI?"
    temperature = 0.2
    max_output_tokens = 100
} | ConvertTo-Json

curl.exe -X POST "http://127.0.0.1:8000/v1/generate" `
  -H "Content-Type: application/json" `
  -d $payload
```

### 5. Health check

```bash
curl http://127.0.0.1:8000/health
```

## Request schema

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `prompt` | string | Yes | Non-empty user prompt |
| `system_instruction` | string | No | System/context message for the LLM |
| `model` | string | No | Model name (defaults to `GEMINI_DEFAULT_MODEL` or `OLLAMA_DEFAULT_MODEL`) |
| `temperature` | float | No | 0.0–2.0 (default: 0.2) |
| `max_output_tokens` | int | No | 1–2048 (default: 512) |
| `top_p` | float | No | 0.0–1.0 (nucleus sampling) |
| `top_k` | int | No | >=1 (top-k sampling) |
| `response_mime_type` | string | No | "text/plain" or "application/json" (default: "text/plain") |

## Response schema

```json
{
  "request_id": "f38558e8-2c31-4c1e-8b59-aaa8f58cd5d3",
  "model": "gemma3",
  "output_text": "FastAPI is a modern web framework...",
  "provider_latency_ms": 842.12,
  "total_latency_ms": 849.34
}
```

## Concurrency & load handling

### Semaphore: concurrent provider calls

`asyncio.Semaphore` limits how many LLM calls can be in-flight at once *per worker process*.

Example with default settings:
- `MAX_CONCURRENT_PROVIDER_CALLS=10` (per worker)
- `--workers 1` (single process)
- **15 concurrent users:**
  - First 10 requests start immediately (~2s latency with Ollama)
  - Next 5 requests queue and wait (~4s latency)
  - **No code changes needed** — requests are handled, just queue briefly

### Tuning for 15+ concurrent users

To eliminate queueing, increase the semaphore in `.env`:

```env
MAX_CONCURRENT_PROVIDER_CALLS=15
```

Or increase workers (production / Linux only):

```bash
# 4 workers × 10 concurrent = 40 total capacity
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Load test numbers

| Metric | Value |
|--------|-------|
| Concurrent users | 15 |
| Semaphore limit (default) | 10 per worker |
| Avg LLM latency | 2–3s |
| Users queued (default) | ~5 |
| Time for all to complete (default) | ~3–5s |
| With `MAX_CONCURRENT_PROVIDER_CALLS=15` | ~2–3s (no queue) |

## LLM backend selection

The app automatically chooses the backend based on `.env`:

```python
if OLLAMA_BASE_URL is set:
    use OllamaService
else:
    use GeminiService
```

### Ollama (local, free)
- **Pros:** No API key needed, runs locally, instant startup, good for development
- **Cons:** Limited model quality (gemma3 vs. Gemini), requires local resources
- **Best for:** Development, prototyping, on-device deployment

### Gemini (cloud, paid)
- **Pros:** High-quality models (Gemini 2.5, etc.), fast inference
- **Cons:** Requires API key, per-token billing, internet connection
- **Best for:** Production, high-quality outputs, scaling

## Docker

```bash
docker compose up --build
```

By default uses Ollama (if running on host as `http://localhost:11434`).
To use Gemini in Docker, set `GEMINI_API_KEY` in `.env` before running.

## Multi-instance load balancing (Nginx)

For production with 50+ concurrent users, use multiple FastAPI instances behind Nginx:

```bash
docker compose -f docker-compose.multi-instance.yml up --build
```

Access via:
```bash
http://127.0.0.1:8080/v1/generate
```

Each instance runs independently with its own semaphore and connection pool.

## Testing

```bash
pytest -q
```

## File layout

```text
app/
  __init__.py
  main.py
  config.py
  schemas.py
  middleware/
    request_id.py
  services/
    gemini.py          # Gemini API service
    ollama.py          # Ollama local service
tests/
  test_health.py
  test_validation.py
deploy/nginx/
  nginx.conf
.github/workflows/
Dockerfile
docker-compose.yml
docker-compose.multi-instance.yml
.env.example
README.md
```

## Troubleshooting

### ModuleNotFoundError: No module named 'app'

**Windows (PowerShell):**
```powershell
# Ensure you're in the repo root and virtualenv is activated
cd D:\Text-2-Text\Text-2-Text
.\venv\Scripts\Activate

# Run with explicit Python interpreter
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

**Linux/Mac:**
```bash
cd /path/to/Text-2-Text
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### OSError: [WinError 10022] An invalid argument was supplied

On Windows, multiprocess workers (`--workers 2+`) cause socket binding errors. **Solution:**
```powershell
# Use --workers 1 or omit --workers flag entirely (defaults to 1)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

### 502 Bad Gateway / Invalid JSON from Ollama

Ollama streams responses as newline-delimited JSON (NDJSON). If you see a 502:
1. Ensure Ollama is running: `ollama serve`
2. Check the gemma3 model is downloaded: `ollama pull gemma3`
3. Verify `.env` has `OLLAMA_BASE_URL=http://localhost:11434`
4. Try a simple request: `curl http://localhost:11434/api/generate -d '{"model":"gemma3","prompt":"hi"}'`

### Connection refused (Ollama not found)

```
HTTPException: Ollama upstream connection failed: Connection refused
```

**Solution:** Start Ollama in a separate terminal:
```bash
ollama serve
```

Or update `.env`:
```env
OLLAMA_BASE_URL=http://10.0.0.2:11434  # If Ollama is on another machine
```

### Timeout (30s) with Gemini

If you consistently hit 30s timeouts with Gemini, increase in `.env`:
```env
REQUEST_TIMEOUT_SECONDS=60
```

## Notes

- **Streaming:** This endpoint returns full responses. For streaming, add an SSE (`/stream`) endpoint.
- **Rate limiting:** For per-user rate limits across instances, add Redis.
- **Monitoring:** Logs include `request_id` and latency metrics; integrate with your observability stack.
- **CORS:** Configured to allow all origins (`*`); tighten in production via `CORS_ALLOW_ORIGINS` in `.env`.

## Contributing

1. Fork the repo
2. Create a feature branch
3. Push and open a PR

## License

MIT
