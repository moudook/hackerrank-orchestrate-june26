import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    def test_init(self):
        rl = TokenBucketRateLimiter(rpm=100, tpm=100000, max_concurrent=3)
        assert rl.rpm == 100
        assert rl.tpm == 100000
        assert rl.max_concurrent == 3
        assert rl.request_tokens == 100
        assert rl.token_tokens == 100000
        assert rl._consecutive_429 == 0
        assert rl._global_cooldown_until == 0.0

    def test_acquire_basic(self):
        rl = TokenBucketRateLimiter(rpm=1000, tpm=1000000)
        start = time.time()
        rl.acquire(estimated_tokens=100)
        elapsed = time.time() - start
        assert elapsed < 0.5
        assert rl.get_current_rpm() >= 1
        assert rl.get_current_tpm() >= 100

    def test_acquire_drains_tokens(self):
        rl = TokenBucketRateLimiter(rpm=1000, tpm=1000000)
        rl.acquire(estimated_tokens=500)
        rl.acquire(estimated_tokens=500)
        assert rl.request_tokens <= 999
        assert rl.token_tokens <= 999000

    def test_stats_structure(self):
        rl = TokenBucketRateLimiter(rpm=90, tpm=900000)
        stats = rl.stats()
        assert 'current_rpm' in stats
        assert 'current_tpm' in stats
        assert 'max_rpm' in stats
        assert 'max_tpm' in stats
        assert 'available_tokens' in stats
        assert 'available_token_tokens' in stats
        assert 'consecutive_429' in stats
        assert 'cooldown_remaining' in stats

    def test_wait_if_needed_no_cooldown(self):
        rl = TokenBucketRateLimiter()
        start = time.time()
        rl.wait_if_needed()
        elapsed = time.time() - start
        assert elapsed < 0.1

    def test_note_429_sets_cooldown(self):
        rl = TokenBucketRateLimiter()
        rl.note_429("Service unavailable")
        assert rl._consecutive_429 == 1
        assert rl._global_cooldown_until > time.time()

    def test_note_429_parses_retry_delay(self):
        rl = TokenBucketRateLimiter()
        msg = '{"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}]}}'
        rl.note_429(msg)
        remaining = rl._global_cooldown_until - time.time()
        assert remaining >= 29.0, f"Expected ~31s cooldown, got {remaining:.1f}s"

    def test_note_429_doubles_on_consecutive(self):
        rl = TokenBucketRateLimiter()
        rl.note_429("Error")
        c1 = rl._global_cooldown_until
        time.sleep(0.01)
        rl.note_429("Error")
        c2 = rl._global_cooldown_until
        assert c2 > c1

    def test_note_success_resets_counter(self):
        rl = TokenBucketRateLimiter()
        rl.note_429("Error")
        rl.note_429("Error")
        assert rl._consecutive_429 == 2
        rl.note_success()
        assert rl._consecutive_429 == 0

    def test_wait_for_cooldown_blocks_when_active(self):
        rl = TokenBucketRateLimiter()
        rl.note_429("retryDelay: 2s")
        start = time.time()
        rl.wait_for_cooldown()
        elapsed = time.time() - start
        assert elapsed >= 1.0

    def test_parse_retry_delay_direct(self):
        rl = TokenBucketRateLimiter()
        delay = rl._parse_retry_delay('retryDelay: "53.686s"')
        assert abs(delay - 53.686) < 0.01

    def test_parse_retry_delay_from_json(self):
        rl = TokenBucketRateLimiter()
        msg = '{"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}]}}'
        delay = rl._parse_retry_delay(msg)
        assert abs(delay - 30.0) < 0.01

    def test_parse_retry_delay_no_match(self):
        rl = TokenBucketRateLimiter()
        delay = rl._parse_retry_delay("no retry info here")
        assert delay == 0.0

    def test_parse_retry_delay_invalid_json(self):
        rl = TokenBucketRateLimiter()
        delay = rl._parse_retry_delay("{invalid json}")
        assert delay == 0.0
