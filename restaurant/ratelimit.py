"""Per-address token buckets: each address gets `per_minute` requests at once and earns them back at that rate.

In process memory, like the sessions (one worker). Behind a reverse proxy the client address must come from the
proxy (`uvicorn --proxy-headers --forwarded-allow-ips=...`), or every request looks like the proxy's.
"""
import math
import threading
from collections import OrderedDict
from time import monotonic


class RateLimiter:
    def __init__(self, per_minute: int, max_addresses: int = 10_000):
        self.capacity = per_minute
        self.rate = per_minute / 60.0                    # tokens per second
        self.max_addresses = max_addresses
        self.buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()   # address -> (tokens, last time)
        self.lock = threading.Lock()

    def allow(self, address: str, now: float | None = None) -> bool:
        now = monotonic() if now is None else now
        with self.lock:
            tokens, last = self.buckets.pop(address, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            allowed = tokens >= 1
            self.buckets[address] = (tokens - 1 if allowed else tokens, now)
            while len(self.buckets) > self.max_addresses:   # the map cannot grow without bound
                self.buckets.popitem(last=False)
            return allowed

    def blocked(self, address: str, now: float | None = None) -> bool:
        """True when the address has no request left now; nothing is used up (for counting only failures)."""
        now = monotonic() if now is None else now
        with self.lock:
            if address not in self.buckets:
                return False
            tokens, last = self.buckets[address]
            return min(self.capacity, tokens + (now - last) * self.rate) < 1

    def retry_after(self) -> int:
        """Seconds until one more request is allowed."""
        return math.ceil(1 / self.rate)

    def reset(self):
        with self.lock:
            self.buckets.clear()
