# FastAPI LLM Async API

Production-style FastAPI starter for async text generation using Gemini (cloud) or Ollama/gemma3 (local).

## What this repo includes

- Dual LLM support: Gemini API or local Ollama
- Async FastAPI endpoint at `/v1/generate`
- Shared async `httpx` client with connection pooling
- `asyncio.Semaphore` to cap concurrent outbound LLM calls
- Request ID middleware for tracing
- Strict Gemini JSON mode with optional response schema enforcement
- Retry/backoff for transient Gemini upstream failures
- Docker and Docker Compose
- Optional Nginx config for multi-instance load balancing

## Architecture

### Single instance

```text
Users -> FastAPI -> Ollama or Gemini API
```

### Multi-instance

```text
Users -> Nginx -> FastAPI 1 / FastAPI 2 / ... -> LLM backend
```

## Why this design

The app does not run the model itself. It focuses on:

- input validation
- request shaping
- async outbound API calls
- concurrency control
- normalized JSON responses
- request tracing

## Quick start

### 1. Clone and configure

```bash
git clone <repo>
cd Text-2-Text
cp .env.example .env
```

### 2a. Use Ollama

Run Ollama locally:

```bash
ollama pull gemma3
ollama serve
```

Then set:

```env
# Leave GEMINI_API_KEY empty or unset
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=gemma3
```

### 2b. Use Gemini

Set:

```env
GEMINI_API_KEY=your_real_api_key_here
GEMINI_DEFAULT_MODEL=gemini-2.5-flash
# Leave OLLAMA_BASE_URL empty or unset to keep Gemini active
```

### 3. Run locally

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Linux or Mac:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 4. Call the API

Minimal request:

```bash
curl -X POST "http://127.0.0.1:8000/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize FastAPI in one paragraph.",
    "temperature": 0.2,
    "max_output_tokens": 200
  }'
```

Gemini text request:

```bash
curl -X POST "http://127.0.0.1:8000/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Translate to technical documentation: Use an async HTTP client.",
    "system_instruction": "Be concise. Use bullet points.",
    "model": "gemini-2.5-flash",
    "temperature": 0.2,
    "max_output_tokens": 300,
    "top_p": 0.95,
    "top_k": 40,
    "response_mime_type": "text/plain"
  }'
```

Strict Gemini JSON mode:

```bash
curl -X POST "http://127.0.0.1:8000/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Return an object with keys status and items.",
    "system_instruction": "Return valid JSON only. No markdown.",
    "model": "gemini-2.5-flash",
    "temperature": 0,
    "max_output_tokens": 512,
    "response_mime_type": "application/json",
    "response_json_schema": {
      "type": "object",
      "properties": {
        "status": { "type": "string" },
        "items": {
          "type": "array",
          "items": { "type": "string" }
        }
      },
      "required": ["status", "items"],
      "additionalProperties": false
    }
  }'
```

Windows PowerShell example:

```powershell
$payload = @'
{
  "prompt": "Return a JSON object with key status and value ok.",
  "system_instruction": "Return valid JSON only. No markdown.",
  "temperature": 0,
  "max_output_tokens": 256,
  "response_mime_type": "application/json"
}
'@

curl.exe -X POST "http://127.0.0.1:8000/v1/generate" `
  -H "Content-Type: application/json" `
  -d $payload
```

### 5. Health check

```bash
curl http://127.0.0.1:8000/health
```

The health response includes:

- `active_backend`
- `gemini_configured`
- `max_concurrent_provider_calls`

## Request schema

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `prompt` | string | Yes | Non-empty user prompt |
| `system_instruction` | string | No | System/context message |
| `model` | string | No | Defaults to provider default model |
| `temperature` | float | No | `0.0` to `2.0` |
| `max_output_tokens` | int | No | `1` to `2048` by default |
| `top_p` | float | No | `0.0` to `1.0` |
| `top_k` | int | No | `>=1` |
| `response_mime_type` | string | No | `"text/plain"` or `"application/json"` |
| `response_json_schema` | object | No | Gemini-only schema for structured JSON output |

`response_json_schema` requires `response_mime_type="application/json"`.

## Response schema

```json
{
  "request_id": "f38558e8-2c31-4c1e-8b59-aaa8f58cd5d3",
  "model": "gemini-2.5-flash",
  "output_text": "{\"status\":\"ok\"}",
  "provider_latency_ms": 842.12,
  "total_latency_ms": 849.34
}
```

When JSON mode is enabled, `output_text` is still returned as a string. That string contains JSON. If `response_json_schema` is provided, the app validates the generated JSON before returning it.

## Concurrency and load handling

`asyncio.Semaphore` limits how many LLM calls can be in flight at once per worker process.

Example:

- `MAX_CONCURRENT_PROVIDER_CALLS=10`
- `--workers 1`
- 15 concurrent users means some requests queue instead of failing

Increase the semaphore in `.env` to reduce queueing:

```env
MAX_CONCURRENT_PROVIDER_CALLS=15
```

Or increase workers on Linux/macOS:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Backend selection

The app chooses the backend from config:

```python
if OLLAMA_BASE_URL is set:
    use OllamaService
else:
    use GeminiService
```

## Gemini wrapper behavior

The Gemini wrapper includes:

- automatic retries for `429`, `500`, `502`, `503`, `504`
- retry handling for timeout and transient network errors
- exponential backoff controlled by config
- optional JSON schema enforcement after generation

Relevant settings:

```env
GEMINI_MAX_RETRIES=2
GEMINI_RETRY_BASE_DELAY_SECONDS=0.5
REQUEST_TIMEOUT_SECONDS=30
```

Supported schema subset for `response_json_schema`:

- `type`
- `properties`
- `required`
- `items`
- `enum`
- `nullable`
- `description`
- `format`

`additionalProperties` is enforced locally by this app, but it is not forwarded to Gemini because Gemini rejects that field in `responseSchema`.

## Docker

```bash
docker compose up --build
```

To use Gemini in Docker, set `GEMINI_API_KEY` in `.env`. To use Ollama, set `OLLAMA_BASE_URL`.

## Deploy on Render

Render docs and pricing:
- https://render.com/pricing
- https://render.com/docs/blueprint-spec
- https://render.com/docs/scaling

This repo includes [render.yaml](D:/Text-2-Text/Text-2-Text/render.yaml) for a non-Docker Python web service on Render.

The current Blueprint is configured for:
- `runtime: python`
- `plan: starter`
- `numInstances: 2`
- `WEB_CONCURRENCY=2`

That means:
- Render runs 2 service instances
- each instance runs 2 Uvicorn workers
- Render handles the load balancer in front of those instances

Use Gemini only on Render for this app:
- set `GEMINI_API_KEY`
- leave `OLLAMA_BASE_URL` unset

### Deploy with `render.yaml`

1. Push this repo to GitHub.
2. In Render, choose `New +` -> `Blueprint`.
3. Connect your GitHub repo.
4. Render will detect `render.yaml`.
5. Add `GEMINI_API_KEY` when prompted.
6. Deploy.

The service starts with:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers $WEB_CONCURRENCY
```

### Manual Render setup

Use these values in the Render dashboard:

- Environment: `Python`
- Plan: `Starter` or higher
- Instance Count: `2`
- Build Command: `pip install -r requirements.txt`
- Start Command: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers $WEB_CONCURRENCY`

Set these environment variables:

```env
APP_ENV=production
APP_HOST=0.0.0.0
WEB_CONCURRENCY=2
GEMINI_API_KEY=your_real_key
GEMINI_DEFAULT_MODEL=gemini-2.5-flash
REQUEST_TIMEOUT_SECONDS=30
MAX_CONCURRENT_PROVIDER_CALLS=10
GEMINI_MAX_RETRIES=2
GEMINI_RETRY_BASE_DELAY_SECONDS=0.5
```

### Notes

- Multi-instance load balancing is not available on Render free web services.
- Render handles load balancing automatically once the service has more than one instance.
- Your public API URL will look like `https://your-service-name.onrender.com`.
- Verify deployment with:

```bash
curl https://your-service-name.onrender.com/health
```

## Multi-instance load balancing

```bash
docker compose -f docker-compose.multi-instance.yml up --build
```

Access via:

```bash
http://127.0.0.1:8080/v1/generate
```

## Testing

```bash
python -m pytest -q
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
    gemini.py
    ollama.py
tests/
  test_config.py
  test_gemini.py
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

### WinError 10022 on Windows

Windows commonly fails with `uvicorn` multiprocess workers. Use a single worker:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Ollama connection refused

Start Ollama:

```bash
ollama serve
```

Or point `OLLAMA_BASE_URL` at the correct host.

### Gemini timeouts

Increase:

```env
REQUEST_TIMEOUT_SECONDS=60
```

### Gemini JSON schema `INVALID_ARGUMENT`

If Gemini rejects the schema, simplify it first to `type`, `properties`, `required`, and `items`. This app already strips unsupported fields before sending the schema upstream.

## Notes

- This endpoint returns full responses, not streaming responses.
- For per-user rate limits across instances, add Redis or another shared store.
- Logs include `request_id` and latency metrics.
- CORS is open by default and should be tightened in production.

## License

MIT
