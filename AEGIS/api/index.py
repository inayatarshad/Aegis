"""Vercel ASGI entrypoint.

Vercel forwards requests under /api/* to this module. The regular development
application remains mounted without changing its local route contract.
"""

import os

os.environ.setdefault("HF_HOME", "/tmp/aegis/huggingface")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aegis/matplotlib")

from fastapi import FastAPI  # noqa: E402

from service.main import app as core_app  # noqa: E402

app = FastAPI(title="AEGIS Vercel Gateway")


@app.get("/")
def gateway_status():
    return {
        "service": "AEGIS Agent Runtime",
        "status": "online",
        "health": "/api/health",
        "docs": "/api/docs",
        "repository": "https://github.com/inayatarshad/Aegis",
    }


app.mount("/api", core_app)
