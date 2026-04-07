import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id and per-request timing headers to each response."""

    async def dispatch(self, request: Request, call_next):
        """Store tracing data on the request and expose it in response headers."""
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        return response
