import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps

_request_id: ContextVar[str] = ContextVar('request_id', default='')


def get_request_id():
    return _request_id.get()


def set_request_id(rid=None):
    rid = rid or uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


class JSONFormatter(logging.Formatter):
    def format(self, record):
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.') + \
             f'{int(record.msecs * 1000):06d}Z'
        log_entry = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }
        if hasattr(record, 'stage'):
            log_entry['stage'] = record.stage
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        if hasattr(record, 'props'):
            log_entry.update(record.props)
        if record.exc_info and record.exc_info[0]:
            log_entry['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def setup_logging(level=logging.INFO, json_output=None):
    if json_output is None:
        json_output = os.getenv('JSON_LOGGING', 'false').lower() in ('true', '1', 'yes')

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if json_output:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)

    return root


def stage_log(logger=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            stage_name = func.__name__
            logger.info(f"Starting stage: {stage_name}", extra={'stage': stage_name})
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    f"Completed stage: {stage_name}",
                    extra={'stage': stage_name, 'duration_ms': round(elapsed_ms, 1)}
                )
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    f"Failed stage: {stage_name}: {e}",
                    extra={'stage': stage_name, 'duration_ms': round(elapsed_ms, 1)}
                )
                raise
        return wrapper
    return decorator
