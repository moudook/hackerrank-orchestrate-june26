import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path)
        self._data = {}
        self._load()

    def _load(self):
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    self._data = json.load(f)
                logger.info(f"Loaded checkpoint with {len(self._data)} completed claims")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                self._data = {}

    def save(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            with open(tmp_path, 'w') as f:
                json.dump(self._data, f, indent=2, default=str)
            tmp_path.replace(self.checkpoint_path)
        except OSError as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def is_processed(self, user_id):
        return user_id in self._data

    def mark_processed(self, user_id, result):
        safe_result = {}
        for k, v in result.items():
            if isinstance(v, (bool, int, float, str)):
                safe_result[k] = v
            elif v is None:
                safe_result[k] = None
            else:
                safe_result[k] = str(v)
        self._data[user_id] = safe_result
        if len(self._data) % 10 == 0:
            self.save()

    def get_completed_count(self):
        return len(self._data)

    def get_completed_ids(self):
        return set(self._data.keys())

    def reset(self):
        self._data = {}
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
