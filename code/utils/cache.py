import hashlib
import json
import logging
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class ResponseCache:
    def __init__(self, cache_dir, enabled=True):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enabled = enabled
        self._memory_cache = {}
        self._hits = 0
        self._misses = 0

        if enabled and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._index_path = self.cache_dir / 'cache_index.json'
            self._index = self._load_index()
        else:
            self._index = {}

    def _load_index(self):
        if self._index_path.exists():
            try:
                with open(self._index_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self):
        if self.cache_dir and self._index:
            try:
                with open(self._index_path, 'w') as f:
                    json.dump(self._index, f)
            except OSError:
                pass

    def _make_key(self, prompt, image_paths, model_name):
        hasher = hashlib.sha256()
        hasher.update(prompt.encode('utf-8'))
        for img_path in sorted(image_paths):
            try:
                stat = os.stat(img_path)
                hasher.update(img_path.encode('utf-8'))
                hasher.update(str(stat.st_size).encode())
                hasher.update(str(stat.st_mtime).encode())
            except OSError:
                hasher.update(img_path.encode('utf-8'))
        hasher.update(model_name.encode('utf-8'))
        return hasher.hexdigest()

    def get(self, prompt, image_paths, model_name):
        if not self.enabled:
            return None

        key = self._make_key(prompt, image_paths, model_name)

        if key in self._memory_cache:
            self._hits += 1
            logger.debug(f"Cache HIT (memory): {key[:12]}")
            return self._memory_cache[key]

        if self.cache_dir:
            cache_file = self.cache_dir / f'{key}.pkl'
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    self._memory_cache[key] = result
                    self._hits += 1
                    logger.debug(f"Cache HIT (disk): {key[:12]}")
                    return result
                except (pickle.UnpicklingError, EOFError, OSError):
                    cache_file.unlink(missing_ok=True)

        self._misses += 1
        return None

    def set(self, prompt, image_paths, model_name, result):
        if not self.enabled:
            return

        key = self._make_key(prompt, image_paths, model_name)
        self._memory_cache[key] = result

        if self.cache_dir:
            cache_file = self.cache_dir / f'{key}.pkl'
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                self._index[key] = {
                    'model': model_name,
                    'num_images': len(image_paths),
                    'prompt_preview': prompt[:60],
                }
                if len(self._index) % 50 == 0:
                    self._save_index()
            except OSError as e:
                logger.warning(f"Failed to write cache: {e}")

    def stats(self):
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate_pct': round(hit_rate, 1),
            'memory_entries': len(self._memory_cache),
            'disk_entries': len(self._index),
        }

    def clear_memory(self):
        self._memory_cache.clear()
