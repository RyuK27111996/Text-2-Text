# FastAPI Gemma API

Minimal FastAPI wrapper for Gemma models served through the Google Gemini API.

## Endpoints

### `POST /v1/text-to-text`

Minimum payload:

```json
{
  "prompt": "Explain FastAPI briefly"
}
```

Optional fields:

- `model`
- `routing_profile`: `auto`, `text_fast`, `text_quality`, `cheap_text`
- `allow_model_fallback`
- `system_instruction`
- `temperature`
- `max_output_tokens`
- `top_p`
- `top_k`
- `response_mime_type`
- `response_json_schema`

### `POST /v1/image-to-text`

Minimum payload:

```json
{
  "prompt": "Describe this image",
  "image_base64": "BASE64_IMAGE_BYTES",
  "image_mime_type": "image/png"
}
```

### `GET /health`

Returns:

```json
{
  "status": "ok",
  "app": "fastapi-gemma-api",
  "env": "dev",
  "active_backend": "gemma",
  "gemini_configured": true,
  "max_concurrent_provider_calls": 10
}
```

## Fallback behavior

Text fallback pools:

- `text_fast`: `gemma-3-27b-it` -> `gemma-3-12b-it` -> `gemma-3-4b-it` -> `gemma-3-2b-it` -> `gemma-3-1b-it`
- `text_quality`: `gemma-3-27b-it` -> `gemma-3-12b-it` -> `gemma-3-4b-it`
- `cheap_text`: `gemma-3-4b-it` -> `gemma-3-2b-it` -> `gemma-3-1b-it`
- `vision_text`: `gemma-3-27b-it` -> `gemma-3-12b-it` -> `gemma-3-4b-it`

On `429`, the current model is cooled down and the request immediately moves to the next compatible Gemma model.

## Local run

Linux or Mac:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Render

The repo includes [render.yaml](D:/Text-2-Text/Text-2-Text/render.yaml) for a non-Docker Python deployment on Render.

Required environment variable:

```env
GEMINI_API_KEY=your_real_key
```

## Testing

```bash
python -m pytest -q
```
