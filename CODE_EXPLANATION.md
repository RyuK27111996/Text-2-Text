# Code Understanding Guide: Text-2-Text FastAPI LLM API

## 🎯 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER                                    │
│                    (15 concurrent)                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP POST /v1/generate
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    FastAPI Application                          │
│                    (Single or Multi-worker)                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Request ID Middleware (Tracing)                        │   │
│  │ CORS Middleware (Cross-origin)                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                      │                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ POST /v1/generate endpoint                             │   │
│  │ - Validate request (Pydantic schema)                   │   │
│  │ - Check semaphore (max 10 concurrent LLM calls)        │   │
│  │ - Call chosen service (Ollama OR Gemini)               │   │
│  │ - Return response with latency metrics                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────┬──────────────────────────┬───────────────────────┘
             │                          │
      ┌──────▼────────┐        ┌────────▼──────┐
      │  OllamaService│        │ GeminiService │
      │  (Local free) │        │  (Cloud paid) │
      └──────┬────────┘        └────────┬──────┘
             │                          │
   ┌─────────▼─────────┐    ┌──────────▼─────────┐
   │ http://localhost: │    │ https://generative │
   │      11434        │    │ language.googleapis│
   │   /api/generate   │    │        .com        │
   │   (gemma3)        │    │ /v1beta/models/... │
   └───────────────────┘    └────────────────────┘
```

---

## 📋 File Structure Explained

### 1. **app/config.py** — Configuration Management

```python
class Settings(BaseSettings):
    # App metadata
    app_name: str = "fastapi-gemini-async-api"
    app_env: str = "dev"
    
    # GEMINI (Cloud) - requires API key
    gemini_api_key: str | None = None                    # Must set in .env
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_default_model: str = "gemma-3-27b-it"
    
    # OLLAMA (Local) - free, runs locally
    ollama_base_url: str = "http://localhost:11434"      # Default: local
    ollama_default_model: str = "gemma3"
    
    # Concurrency control
    max_concurrent_provider_calls: int = 10               # Semaphore limit
    
    # Request limits
    max_input_chars: int = 16000                          # Max prompt length
    max_output_tokens_limit: int = 2048                   # Max response tokens
    default_max_output_tokens: int = 512                  # Default response size
    request_timeout_seconds: float = 30.0                 # HTTP timeout
    
    # CORS
    cors_allow_origins: list[str] = ["*"]                # Allow all origins
```

**How it works:**
- Reads from `.env` file automatically (e.g., `GEMINI_API_KEY=xxx`)
- `@lru_cache` stores one instance in memory (singleton pattern)
- All settings accessible via `get_settings()`

---

### 2. **app/schemas.py** — Data Validation (Pydantic)

```python
# REQUEST: What users send to /v1/generate
class GenerateRequest(BaseModel):
    prompt: str                    # Required: "Summarize FastAPI"
    system_instruction: str | None # Optional: "Be concise"
    model: str | None              # Optional: defaults to config model
    temperature: float = 0.2       # 0.0=deterministic, 2.0=creative
    max_output_tokens: int | None  # How long the response can be
    top_p: float | None            # Nucleus sampling (0-1)
    top_k: int | None              # Top-K sampling (>=1)
    response_mime_type: str = "text/plain"  # "text/plain" or "application/json"
    
    # Validators: Custom rules
    @field_validator("prompt")
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt cannot be empty")
        return value.strip()

# RESPONSE: What the server returns
class GenerateResponse(BaseModel):
    request_id: str                # Trace ID for logging
    model: str                     # Which model was used
    output_text: str               # The LLM's response
    provider_latency_ms: float     # Time spent calling Gemini/Ollama
    total_latency_ms: float        # Total request time (including validation, etc.)
```

**How Pydantic works:**
- Automatically validates incoming JSON against the schema
- Rejects invalid data (e.g., negative `temperature`)
- Converts types (e.g., string "0.5" → float 0.5)
- If validation fails, returns `422 Unprocessable Entity`

---

### 3. **app/middleware/request_id.py** — Request Tracing

```python
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. Extract or generate request ID
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        #    ↑ Either use the one sent by client or create a new UUID
        
        # 2. Store in request.state for handlers to access
        request.state.request_id = request_id
        
        # 3. Time the request
        start = time.perf_counter()
        response = await call_next(request)  # Call the actual endpoint
        
        # 4. Add timing and request ID to response headers
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        
        return response
```

**Why this matters:**
- Trace requests across logs (every log message includes the request_id)
- Measure total latency from the client's perspective
- Correlate requests across microservices

---

### 4. **app/services/gemini.py** — Gemini LLM Integration

```python
class GeminiService:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, semaphore: asyncio.Semaphore):
        self.settings = settings      # Configuration
        self.client = client          # Shared async HTTP client (connection pooling)
        self.semaphore = semaphore    # Limits concurrent requests to max 10

    async def generate(self, payload: GenerateRequest) -> GeminiResult:
        # 1. Validate API key is set
        if not self.settings.gemini_api_key:
            raise HTTPException(500, "GEMINI_API_KEY is not configured")
        
        # 2. Choose model (use user's choice or default)
        model = payload.model or self.settings.gemini_default_model
        
        # 3. Prepare request body (Gemini API format)
        body = {
            "contents": [{
                "role": "user",
                "parts": [{"text": payload.prompt}]
            }],
            "generationConfig": {
                "temperature": payload.temperature,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": payload.response_mime_type,
                "topP": payload.top_p,  # Optional
                "topK": payload.top_k,  # Optional
            },
            "systemInstruction": {"parts": [{"text": payload.system_instruction}]}
        }
        
        # 4. Build endpoint URL
        endpoint = f"{base_url}/{api_version}/models/{model}:generateContent"
        # Example: https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent
        
        # 5. Acquire semaphore slot (wait if 10 requests already in-flight)
        async with self.semaphore:
            # If queue exists, wait here
            response = await self.client.post(endpoint, json=body, headers={...})
        
        # 6. Parse response and extract text
        data = response.json()
        text = self._extract_text(data)  # Navigate nested JSON
        
        return GeminiResult(model=model, text=text, provider_latency_ms=...)
    
    @staticmethod
    def _extract_text(data: dict) -> str:
        # Gemini response structure:
        # {
        #   "candidates": [{
        #     "content": {
        #       "parts": [{"text": "The response..."}]
        #     }
        #   }]
        # }
        candidates = data.get("candidates") or []
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts = [part.get("text") for part in parts if part.get("text")]
        return "\n".join(texts).strip()
```

**Key concepts:**
- **Async**: Non-blocking I/O (doesn't wait synchronously)
- **Semaphore**: Acts as a bouncer — only 10 requests can call Gemini simultaneously
- **Error handling**: Catches timeouts, connection errors, API errors

---

### 5. **app/services/ollama.py** — Local LLM Integration

```python
class OllamaService:
    # Same interface as GeminiService!
    async def generate(self, payload: GenerateRequest) -> OllamaResult:
        # 1. Choose model
        model = payload.model or self.settings.ollama_default_model  # "gemma3"
        
        # 2. Prepare Ollama API request (simpler than Gemini)
        body = {
            "model": model,
            "prompt": payload.prompt,
            "temperature": payload.temperature,
            "max_tokens": max_output_tokens,
        }
        
        # 3. Call Ollama
        endpoint = f"{self.settings.ollama_base_url}/api/generate"
        # Example: http://localhost:11434/api/generate
        
        async with self.semaphore:
            response = await self.client.post(endpoint, json=body)
        
        # 4. **KEY DIFFERENCE**: Ollama streams as NDJSON (newline-delimited JSON)
        # Each line is a JSON object:
        # {"model":"gemma3","response":"Hi","done":false}
        # {"model":"gemma3","response":" there","done":false}
        # {"model":"gemma3","response":"!","done":true}
        
        # Parse NDJSON and concatenate "response" fields
        raw = response.text
        parts = []
        for line in raw.splitlines():
            obj = json.loads(line)
            if "response" in obj:
                parts.append(obj["response"])
        
        text = "".join(parts)
        return OllamaResult(model=model, text=text.strip(), ...)
```

**Why two services?**
- Same interface (`async generate()` method) → easy to swap
- Different API formats (Gemini JSON vs. Ollama NDJSON)
- Different backends (cloud vs. local)

---

### 6. **app/main.py** — Application Entry Point

```python
# 1. LOGGING SETUP
logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("app")

# 2. CUSTOM LOGGER ADAPTER (adds request_id to every log message)
class RequestLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # Every log gets: "2026-04-07 02:14:26 INFO app [request_id=abc123] Message"
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("request_id", self.extra.get("request_id", "-"))
        return msg, kwargs

# 3. APPLICATION LIFECYCLE (@asynccontextmanager — startup/shutdown hook)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP (runs once when server starts)
    settings = get_settings()
    
    # Create shared async HTTP client (reuses connections)
    # max_keepalive_connections=100: Keep 100 idle connections alive
    # max_connections=200: Allow up to 200 total connections
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.request_timeout_seconds),
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=200)
    )
    
    # Create semaphore (only 10 concurrent LLM calls)
    app.state.provider_semaphore = asyncio.Semaphore(
        settings.max_concurrent_provider_calls  # 10
    )
    
    logger.info("Application startup complete", extra={"request_id": "startup"})
    
    yield  # ← Server runs here (handles requests)
    
    # SHUTDOWN (runs once when server stops)
    await app.state.http_client.aclose()
    logger.info("Application shutdown complete", extra={"request_id": "shutdown"})

# 4. CREATE FASTAPI APP
app = FastAPI(
    title="FastAPI Gemini Async API",
    version="0.1.0",
    lifespan=lifespan,  # Register startup/shutdown hooks
)

# 5. ADD MIDDLEWARE
app.add_middleware(RequestContextMiddleware)  # Tracing
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)  # Cross-origin

# 6. HEALTH CHECK ENDPOINT
@app.get("/health")
async def health(request: Request):
    # Returns app status and config
    return {
        "status": "ok",
        "app": settings.app_name,
        "gemini_configured": bool(settings.gemini_api_key),
        "max_concurrent_provider_calls": 10,
    }

# 7. MAIN ENDPOINT: POST /v1/generate
@app.post(
    "/v1/generate",
    response_model=GenerateResponse,  # Pydantic validates response
    responses={429: {...}, 500: {...}, 502: {...}, 504: {...}},  # Possible errors
)
async def generate(request: Request, payload: GenerateRequest):
    # Extract request ID from middleware
    request_id = getattr(request.state, "request_id", "-")
    log = RequestLoggerAdapter(logger, {"request_id": request_id})
    
    # Validate prompt length
    if len(payload.prompt) > settings.max_input_chars:
        raise HTTPException(422, "prompt exceeds max characters")
    
    # CHOOSE SERVICE: Ollama (local) or Gemini (cloud)
    if settings.ollama_base_url:
        service = OllamaService(settings, app.state.http_client, app.state.provider_semaphore)
    else:
        service = GeminiService(settings, app.state.http_client, app.state.provider_semaphore)
    
    # CALL LLM
    total_started = time.perf_counter()
    log.info("Incoming generation request")
    result = await service.generate(payload)
    total_latency_ms = (time.perf_counter() - total_started) * 1000.0
    log.info("Generation request completed")
    
    # RETURN RESPONSE
    return GenerateResponse(
        request_id=request_id,
        model=result.model,
        output_text=result.text,
        provider_latency_ms=result.provider_latency_ms,
        total_latency_ms=total_latency_ms,
    )
```

---

## 🔄 Request Flow (Example)

**User sends:**
```json
POST /v1/generate HTTP/1.1
Content-Type: application/json

{
  "prompt": "What is async programming?",
  "temperature": 0.2,
  "max_output_tokens": 100
}
```

**Behind the scenes:**

```
1. Request arrives
   ↓
2. RequestContextMiddleware
   - Generate request_id = "abc-123"
   - Store in request.state.request_id
   ↓
3. Pydantic validates GenerateRequest
   - prompt is not empty ✓
   - temperature is 0.0-2.0 ✓
   - max_output_tokens is 1-2048 ✓
   ↓
4. Handler: async def generate(...)
   - Extract request_id = "abc-123"
   - Choose service (check ollama_base_url)
   ↓
5. Service.generate(payload)
   - Try to acquire semaphore slot
   - Wait if 10 other requests are in-flight
   - Build API request
   - Call Ollama or Gemini
   - Parse response
   - Return result
   ↓
6. Create GenerateResponse
   - Include request_id, model, text, latencies
   ↓
7. Pydantic validates response schema
   ↓
8. FastAPI converts to JSON
   ↓
9. RequestContextMiddleware adds headers
   - x-request-id: abc-123
   - x-process-time-ms: 2345.67
   ↓
10. Return to user
```

**User receives:**
```json
HTTP/1.1 200 OK
x-request-id: abc-123
x-process-time-ms: 2345.67
Content-Type: application/json

{
  "request_id": "abc-123",
  "model": "gemma3",
  "output_text": "Async programming allows your program to handle multiple tasks concurrently...",
  "provider_latency_ms": 1234.56,
  "total_latency_ms": 2345.67
}
```

---

## 📊 Concurrency Model

### Semaphore (Bottleneck)

```
15 concurrent users sending requests
│
├─ User 1: Acquires semaphore slot → Calls Ollama
├─ User 2: Acquires semaphore slot → Calls Ollama
├─ User 3: Acquires semaphore slot → Calls Ollama
├─ ...
├─ User 10: Acquires semaphore slot → Calls Ollama
│
└─ User 11-15: WAIT (blocked on semaphore)
   └─ When User 1 finishes (2-3s later), User 11 acquires slot
```

**Why?**
- Semaphore(10) = only 10 concurrent outbound requests to LLM
- 15 users → 5 must queue
- Users 11-15 experience ~2-3s extra latency

**Solution:** Set `MAX_CONCURRENT_PROVIDER_CALLS=15` in `.env`

---

## 🎓 Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Async** | Non-blocking I/O. Request handler waits for I/O (Ollama/Gemini) without blocking other requests. |
| **Semaphore** | Token-based access control. Only N concurrent operations allowed. Others wait. |
| **Middleware** | Intercepts every request/response. Used for logging, CORS, tracing. |
| **Pydantic** | Data validation library. Validates JSON input/output against schemas. |
| **httpx.AsyncClient** | Async HTTP client. Reuses connections (better than creating new ones). |
| **@asynccontextmanager** | Startup/shutdown hooks. Setup resources on start, cleanup on stop. |
| **Request ID** | Unique identifier for each request. Traced through logs for debugging. |

---

## 🚀 To Run This Code

**Start Ollama (local):**
```bash
ollama serve
```

**Start the API (PowerShell):**
```powershell
.\venv\Scripts\Activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

**Send a request:**
```bash
curl -X POST "http://127.0.0.1:8000/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hi","temperature":0.2,"max_output_tokens":50}'
```

---

**Questions?** This codebase is designed to be:
✅ Async (fast, non-blocking)
✅ Scalable (semaphore + multiple workers)
✅ Observable (request IDs + latency metrics)
✅ Flexible (supports both cloud & local LLMs)
