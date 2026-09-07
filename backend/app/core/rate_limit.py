"""
Minimal in-memory sliding-window rate limiter for sensitive, unauthenticated
auth endpoints (login, signup, forgot-password) - the ones a brute-force or
account-enumeration script would hit. No external dependency (e.g. redis);
this is process-local, which is fine for the current single-process
deployment model. If this app is ever run as multiple worker processes
behind a load balancer, replace this with a shared store (Redis) - the
counters here would otherwise be per-process and under-count.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, List

from fastapi import HTTPException, Request, status

_lock = threading.Lock()
_hits: Dict[str, List[float]] = defaultdict(list)


def rate_limit(max_requests: int, window_seconds: float):
    """Dependency factory. Usage: Depends(rate_limit(5, 60)) - at most 5
    requests per client IP per 60-second sliding window for this route."""

    def _check(request: Request):
        client_ip = request.client.host if request.client else "unknown"
        key = f"{request.url.path}:{client_ip}"
        now = time.monotonic()

        with _lock:
            hits = _hits[key]
            cutoff = now - window_seconds
            while hits and hits[0] < cutoff:
                hits.pop(0)

            if len(hits) >= max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please try again later.",
                )

            hits.append(now)

    return _check
