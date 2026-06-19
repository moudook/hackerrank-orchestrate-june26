import os
import sys
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.rate_limiter import AdaptiveRateLimiter


class TestAdaptiveRateLimiter:
    def test_init(self):
        rl = AdaptiveRateLimiter(max_concurrent=3)
        assert rl.stats()['consecutive_429'] == 0
        assert rl.stats()['cooldown_remaining'] == 0.0

    def test_acquire_basic(self):
        rl = AdaptiveRateLimiter(max_concurrent=10)
        start = time.time()
        rl.acquire()
        rl.release()
        elapsed = time.time() - start
        assert elapsed < 1.0

    def test_stats_structure(self):
        rl = AdaptiveRateLimiter()
        stats = rl.stats()
        assert 'current_rpm' in stats
        assert 'consecutive_429' in stats
        assert 'cooldown_remaining' in stats

    def test_wait_if_needed_no_cooldown(self):
        rl = AdaptiveRateLimiter()
        start = time.time()
        rl.wait_if_needed()
        elapsed = time.time() - start
        assert elapsed < 0.1

    def test_note_429_sets_cooldown(self):
        rl = AdaptiveRateLimiter()
        rl.note_429("Service unavailable")
        stats = rl.stats()
        assert stats['consecutive_429'] == 1
        assert stats['cooldown_remaining'] > 0.0

    def test_note_429_parses_retry_delay(self):
        rl = AdaptiveRateLimiter()
        msg = '{"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}]}}'
        rl.note_429(msg)
        remaining = rl.stats()['cooldown_remaining']
        assert remaining >= 29.0, f"Expected ~31s cooldown, got {remaining:.1f}s"

    def test_note_429_doubles_on_consecutive(self):
        rl = AdaptiveRateLimiter()
        rl.note_429("Error")
        c1 = rl.stats()['cooldown_remaining']
        time.sleep(0.01)
        rl.note_429("Error")
        c2 = rl.stats()['cooldown_remaining']
        assert c2 > c1

    def test_note_success_reduces_counter(self):
        rl = AdaptiveRateLimiter()
        rl.note_429("Error")
        rl.note_429("Error")
        assert rl.stats()['consecutive_429'] == 2
        rl.note_success()
        stats = rl.stats()
        assert stats['consecutive_429'] == 1

    def test_wait_for_cooldown_blocks_when_active(self):
        rl = AdaptiveRateLimiter()
        rl.note_429("retryDelay: 2s")
        start = time.time()
        rl.wait_for_cooldown()
        elapsed = time.time() - start
        assert elapsed >= 1.0

    def test_concurrent_acquire_blocks_at_max(self):
        rl = AdaptiveRateLimiter(max_concurrent=2)
        acquired = []

        def grab():
            rl.acquire()
            acquired.append(threading.current_thread().name)
            time.sleep(0.3)  # hold the slot
            rl.release()

        threads = [threading.Thread(target=grab, name=f"t{i}") for i in range(4)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start
        assert len(acquired) == 4
        # With max_concurrent=2 and 0.3s each, should take at least 0.6s
        assert elapsed >= 0.5

    def test_parse_retry_delay_direct(self):
        rl = AdaptiveRateLimiter()
        # Use note_429 to trigger parsing
        rl.note_429('retryDelay: "53.686s"')
        remaining = rl.stats()['cooldown_remaining']
        assert remaining >= 53.0, f"Expected ~54.6s cooldown, got {remaining:.1f}s"

    def test_parse_retry_delay_from_json(self):
        rl = AdaptiveRateLimiter()
        msg = '{"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}]}}'
        rl.note_429(msg)
        remaining = rl.stats()['cooldown_remaining']
        assert remaining >= 29.0, f"Expected ~31s cooldown, got {remaining:.1f}s"

    def test_parse_retry_delay_no_match(self):
        rl = AdaptiveRateLimiter()
        rl.note_429("no retry info here")
        cooldown = rl.stats()['cooldown_remaining']
        # Exponential backoff: 2^1 = 2s, so cooldown should be ~2s
        assert cooldown >= 1.0, f"Expected ~2s cooldown, got {cooldown:.1f}s"

    def test_parse_retry_delay_invalid_json(self):
        rl = AdaptiveRateLimiter()
        rl.note_429("{invalid json}")
        cooldown = rl.stats()['cooldown_remaining']
        # Should fall back to exponential: 2^1 = 2s
        assert cooldown >= 1.0

    def test_start_with_zero_delay(self):
        rl = AdaptiveRateLimiter()
        # Before any 429, acquire should be instant
        start = time.time()
        rl.acquire()
        rl.release()
        elapsed = time.time() - start
        assert elapsed < 0.1, f"Expected near-zero delay, got {elapsed:.3f}s"

    def test_adapts_to_429_then_recovers(self):
        rl = AdaptiveRateLimiter()
        # Simulate burst of 429s
        for i in range(3):
            rl.note_429(f"Error {i}")
            time.sleep(0.01)
        c3 = rl.stats()['consecutive_429']
        assert c3 == 3
        # Simulate success recovery
        for i in range(3):
            rl.note_success()
        assert rl.stats()['consecutive_429'] == 0

    def test_get_current_rpm(self):
        rl = AdaptiveRateLimiter()
        for _ in range(3):
            rl.acquire()
            rl.release()
        assert rl.get_current_rpm() >= 3

    def test_inter_call_gap_on_429(self):
        rl = AdaptiveRateLimiter(max_concurrent=5)
        rl.note_429("Error")  # sets 2s cooldown + consecutive=1
        # First acquire waits for the 2s cooldown
        t0 = time.time()
        rl.acquire()
        rl.release()
        first_call = time.time() - t0
        assert first_call >= 1.5
        # Second acquire: cooldown already expired from first wait,
        # but inter-call gap of 2s (2^1) between the two calls
        t1 = time.time()
        rl.acquire()
        rl.release()
        second_call = time.time() - t1
        assert second_call >= 1.5

    def test_no_inter_call_gap_when_no_429(self):
        rl = AdaptiveRateLimiter()
        t0 = time.time()
        rl.acquire()
        rl.release()
        rl.acquire()
        rl.release()
        t1 = time.time()
        # Without any 429, both acquires should be instant
        assert t1 - t0 < 0.2
