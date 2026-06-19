import time
import logging
import threading

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, call_delay=0.6, batch_size=5, batch_pause=10.0):
        self.call_delay = call_delay
        self.batch_size = batch_size
        self.batch_pause = batch_pause
        self.call_count = 0
        self.lock = threading.Lock()

    def wait_if_needed(self):
        sleep_time = 0
        with self.lock:
            if self.call_count > 0 and self.call_count % self.batch_size == 0:
                sleep_time = self.batch_pause
            elif self.call_count > 0:
                sleep_time = self.call_delay
            self.call_count += 1
            
        if sleep_time > 0:
            if sleep_time == self.batch_pause:
                logger.info(f"Rate limit: batch of {self.batch_size} complete, pausing {self.batch_pause}s")
            time.sleep(sleep_time)
