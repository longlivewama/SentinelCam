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

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# deque, not list: expiring old hits pops from the FRONT, which is O(n)
# on a list and O(1) here. Entries are dropped once a key's window
# empties (see below) - keeping them would mean one permanently retained
# entry per (path, client IP) ever seen, i.e. unbounded memory growth
# driven entirely by unauthenticated traffic.
_hits: Dict[str, Deque[float]] = defaultdict(deque)


def rate_limit(max_requests: int, window_seconds: float):
    """Dependency factory. Usage: Depends(rate_limit(5, 60)) - at most 5
    requests per client IP per 60-second sliding window for this route."""

    def _check(request: Request):
        # request.client.host is the immediate peer. Behind a reverse
        # proxy or load balancer that is the proxy, so every client shares
        # one bucket - deploy with the proxy configured to preserve the
        # client address (and this app behind a trusted-proxy middleware)
        # before relying on these limits in production. See SECURITY.md.
        client_ip = request.client.host if request.client else "unknown"
        key = f"{request.url.path}:{client_ip}"
        now = time.monotonic()

        with _lock:
            hits = _hits[key]
            cutoff = now - window_seconds
            while hits and hits[0] < cutoff:
                hits.popleft()

            if len(hits) >= max_requests:
                # Leave the key in place: it still holds live hits, and
                # dropping it here would reset the window on every
                # rejection - turning the limiter off exactly when it is
                # being exercised.
                logger.warning(
                    "Rate limit hit: %s from %s (%d requests in %.0fs)",
                    request.url.path, client_ip, max_requests, window_seconds,
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please try again later.",
                )

            hits.append(now)

    return _check


def prune_expired(max_window_seconds: float = 3600.0):
    """Drops keys whose most recent hit is older than any window in use.
    Called from the periodic maintenance task in main.py so a long-running
    process doesn't accumulate an entry per client IP indefinitely."""
    now = time.monotonic()
    with _lock:
        stale = [key for key, hits in _hits.items() if not hits or (now - hits[-1]) > max_window_seconds]
        for key in stale:
            del _hits[key]
    return len(stale)
