import json
import logging
import re
import threading
import time

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    def __init__(self, rpm=90, tpm=900000, max_concurrent=5):
        self.rpm = rpm
        self.tpm = tpm
        self.max_concurrent = max_concurrent

        self.request_tokens = rpm
        self.token_tokens = tpm
        self.last_refill = time.time()

        self.semaphore = threading.Semaphore(max_concurrent)
        self.lock = threading.Lock()
        self.call_times: list[float] = []

        self._global_cooldown_until = 0.0
        self._consecutive_429 = 0

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.last_refill = now

        self.request_tokens = min(self.rpm, self.request_tokens + elapsed * (self.rpm / 60.0))
        self.token_tokens = min(self.tpm, self.token_tokens + elapsed * (self.tpm / 60.0))

    def _parse_retry_delay(self, error_message: str) -> float:
        try:
            m = re.search(r'retryDelay["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)s', error_message)
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

    def note_429(self, error_message: str) -> float:
        with self.lock:
            self._consecutive_429 += 1
            delay = self._parse_retry_delay(error_message)
            if delay > 0:
                wait = delay + 1.0
            else:
                wait = min(2 ** self._consecutive_429, 120.0)
            cooldown_until = time.time() + wait
            if cooldown_until > self._global_cooldown_until:
                self._global_cooldown_until = cooldown_until
            logger.warning(f"429 rate limit: backing off {wait:.1f}s (consecutive={self._consecutive_429})")
            return wait

    def note_success(self):
        with self.lock:
            self._consecutive_429 = 0

    def wait_for_cooldown(self):
        remaining = self._global_cooldown_until - time.time()
        if remaining > 0:
            logger.info(f"Global cooldown active: waiting {remaining:.0f}s")
            time.sleep(remaining)

    def acquire(self, estimated_tokens=0):
        with self.semaphore:
            with self.lock:
                self.wait_for_cooldown()
                self._refill()

                if self.request_tokens < 1 or self.token_tokens < estimated_tokens:
                    wait_time = max(
                        (1 - self.request_tokens) / (self.rpm / 60.0),
                        (estimated_tokens - self.token_tokens) / (self.tpm / 60.0),
                        0.5
                    )
                    logger.debug(f"Rate limit wait: {wait_time:.1f}s "
                                 f"(req_tokens={self.request_tokens:.1f}, tok_tokens={self.token_tokens:.1f})")
                    time.sleep(wait_time)
                    self._refill()

                self.request_tokens -= 1
                self.token_tokens -= estimated_tokens

                now = time.time()
                cutoff = now - 60
                self.call_times = [t for t in self.call_times if t > cutoff]
                self.call_times.append(now)

    def wait_if_needed(self):
        self.wait_for_cooldown()

    def get_current_rpm(self):
        cutoff = time.time() - 60
        return len([t for t in self.call_times if t > cutoff])

    def get_current_tpm(self):
        return int(self.tpm - self.token_tokens)

    def stats(self):
        return {
            'current_rpm': self.get_current_rpm(),
            'current_tpm': self.get_current_tpm(),
            'max_rpm': self.rpm,
            'max_tpm': self.tpm,
            'available_tokens': round(self.request_tokens, 1),
            'available_token_tokens': int(self.token_tokens),
            'consecutive_429': self._consecutive_429,
            'cooldown_remaining': max(0.0, round(self._global_cooldown_until - time.time(), 1)),
        }
