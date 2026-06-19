import json
import logging
import re
import threading
import time

logger = logging.getLogger(__name__)


class AdaptiveRateLimiter:
    """Rate limiter with adaptive exponential backoff based on API responses.

    - Starts with zero delay; ramps up only when 429s are received.
    - On 429: exponential backoff (1s, 2s, 4s, ... capped at 120s).
    - On success: gradual recovery (decrements backoff counter).
    - Server retry-after headers take priority over exponential backoff.
    - Thread-safe for parallel workers via semaphore + lock.

    Usage:
        rate_limiter = AdaptiveRateLimiter(max_concurrent=10)
        rate_limiter.acquire()       # blocks until slot + cooldown available
        try:
            result = api_call(...)
            rate_limiter.note_success()
        except RateLimitError:
            rate_limiter.note_429(str(e))
            raise
        finally:
            rate_limiter.release()   # frees slot for next worker
    """

    def __init__(self, max_concurrent=10):
        self._lock = threading.Lock()
        self._api_semaphore = threading.Semaphore(max_concurrent)

        self._global_cooldown_until = 0.0
        self._consecutive_429 = 0
        self._call_times: list[float] = []
        self._last_acquire_time: float = float('-inf')

    def _parse_retry_delay(self, error_message: str) -> float:
        try:
            m = re.search(r'(?:retryAfter|retryDelay)["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)s', error_message)
            if m:
                return float(m.group(1))
            data = json.loads(error_message)
            details = data.get('error', {}).get('details', [])
            for d in details:
                if d.get('@type', '').endswith('RetryInfo'):
                    rd = d.get('retryDelay', '0s')
                    m = re.search(r'(\d+(?:\.\d+)?)s', rd)
                    if m:
                        return float(m.group(1))
        except (json.JSONDecodeError, AttributeError, KeyError, ValueError):
            pass
        return 0.0

    def note_429(self, error_message: str = '') -> float:
        with self._lock:
            self._consecutive_429 += 1
            delay = self._parse_retry_delay(error_message)
            if delay <= 0:
                delay = min(2 ** self._consecutive_429, 120.0)
            cooldown = time.time() + delay
            if cooldown > self._global_cooldown_until:
                self._global_cooldown_until = cooldown
            logger.warning(f"429 rate limit: backing off {delay:.1f}s (consecutive={self._consecutive_429})")
            return delay

    def note_success(self):
        with self._lock:
            if self._consecutive_429 > 0:
                self._consecutive_429 -= 1

    def acquire(self):
        """Acquire permission to make an API call.

        Blocks until a concurrent slot is available, any global
        cooldown has expired, and the adaptive inter-call gap is met.

        Under zero rate-limit pressure the gap is 0 → full parallelism.
        Under pressure the gap grows exponentially, serialising calls
        so the API can recover between requests.
        """
        self._api_semaphore.acquire()
        while True:
            remaining = self._global_cooldown_until - time.time()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.5))
        with self._lock:
            now = time.time()
            if self._consecutive_429 > 0:
                gap = min(2 ** self._consecutive_429, 60.0)
                since_last = now - self._last_acquire_time
                if since_last < gap:
                    time.sleep(gap - since_last)
                    now = time.time()
            self._last_acquire_time = now
            cutoff = now - 60
            self._call_times = [t for t in self._call_times if t > cutoff]
            self._call_times.append(now)

    def release(self):
        """Release the concurrent slot after an API call completes."""
        self._api_semaphore.release()

    def wait_for_cooldown(self):
        """Block until global cooldown expires (if any)."""
        while True:
            remaining = self._global_cooldown_until - time.time()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 1.0))

    def wait_if_needed(self):
        self.wait_for_cooldown()

    def get_current_rpm(self) -> int:
        cutoff = time.time() - 60
        return len([t for t in self._call_times if t > cutoff])

    def stats(self) -> dict:
        return {
            'current_rpm': self.get_current_rpm(),
            'consecutive_429': self._consecutive_429,
            'cooldown_remaining': max(0.0, round(self._global_cooldown_until - time.time(), 1)),
        }
