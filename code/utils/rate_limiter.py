import logging
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
        self.call_times = []

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.last_refill = now

        self.request_tokens = min(self.rpm, self.request_tokens + elapsed * (self.rpm / 60.0))
        self.token_tokens = min(self.tpm, self.token_tokens + elapsed * (self.tpm / 60.0))

    def acquire(self, estimated_tokens=0):
        with self.semaphore:
            with self.lock:
                self._refill()

                if self.request_tokens < 1 or self.token_tokens < estimated_tokens:
                    wait_time = max(
                        (1 - self.request_tokens) / (self.rpm / 60.0),
                        (estimated_tokens - self.token_tokens) / (self.tpm / 60.0),
                        0.5
                    )
                    logger.debug(f"Rate limit wait: {wait_time:.1f}s (req_tokens={self.request_tokens:.1f}, tok_tokens={self.token_tokens:.1f})")
                    time.sleep(wait_time)
                    self._refill()

                self.request_tokens -= 1
                self.token_tokens -= estimated_tokens

                now = time.time()
                cutoff = now - 60
                self.call_times = [t for t in self.call_times if t > cutoff]
                self.call_times.append(now)

    def wait_if_needed(self):
        pass

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
        }
