"""Vercel serverless entrypoint.

Exposes the existing FastAPI application as the Vercel `app` while mapping
Vercel's `/api/*` request paths back onto the backend's native routes
(`/api/predict` -> `/predict`, `/api/health` -> `/health`).

No prediction logic lives here; `app.backend.main` remains the single source
of truth. This wrapper is pure path-routing so the same binary serves both the
Vercel prefix and a local run without duplicating routes.

Deploy shape (vercel.json):
  - /api/*            -> this function
  - / , /predict , /health -> this function (served by the backend routes)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend.main import app as backend_app  # noqa: E402

_API_PREFIX = "/api"


class StripApiPrefix:
    """ASGI middleware that rewrites a leading `/api` path prefix onto the backend app.

    Vercel invokes Python functions for routes under `/api/*` but passes the full
    clean URL path (e.g. `/api/predict`). FastAPI declares routes without the
    prefix, so we rewrite the scope path back to the backend's native form.
    """

    def __init__(self, next_app):
        self.next_app = next_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            # Delegate startup/shutdown to the backend app so the model is
            # preloaded once per warm instance (better serverless cold start).
            await self.next_app(scope, receive, send)
            return
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path.startswith(_API_PREFIX):
                stripped = path[len(_API_PREFIX):] or "/"
                scope["path"] = stripped
                raw = scope.get("raw_path")
                if raw:
                    scope["raw_path"] = stripped.encode("latin-1")
                if not scope.get("root_path"):
                    scope["root_path"] = _API_PREFIX
        await self.next_app(scope, receive, send)


app = StripApiPrefix(backend_app)