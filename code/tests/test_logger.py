import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import logging
import pytest
from io import StringIO

from utils.logger import JSONFormatter, get_request_id, set_request_id, stage_log, setup_logging


class TestRequestID:
    def test_default_is_empty(self):
        assert get_request_id() == ''

    def test_set_and_get(self):
        rid = set_request_id('test-123')
        assert get_request_id() == 'test-123'
        assert rid == 'test-123'

    def test_auto_generate(self):
        rid = set_request_id()
        assert len(rid) == 12

    def test_overwrite(self):
        set_request_id('first')
        set_request_id('second')
        assert get_request_id() == 'second'


class TestJSONFormatter:
    def test_basic_format(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test_logger', level=logging.INFO,
            pathname='', lineno=0, msg='hello world',
            args=(), exc_info=None
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed['message'] == 'hello world'
        assert parsed['level'] == 'INFO'
        assert parsed['logger'] == 'test_logger'
        assert 'timestamp' in parsed

    def test_format_with_stage(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test', level=logging.INFO,
            pathname='', lineno=0, msg='processing',
            args=(), exc_info=None
        )
        record.stage = 'preprocessor'
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed['stage'] == 'preprocessor'

    def test_format_with_duration(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test', level=logging.INFO,
            pathname='', lineno=0, msg='done',
            args=(), exc_info=None
        )
        record.duration_ms = 123.4
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed['duration_ms'] == 123.4

    def test_format_with_props(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test', level=logging.INFO,
            pathname='', lineno=0, msg='test',
            args=(), exc_info=None
        )
        record.props = {'user_id': 'u123', 'tokens': 500}
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed['user_id'] == 'u123'
        assert parsed['tokens'] == 500


class TestSetupLogging:
    def test_plain_format_default(self):
        logger = setup_logging(json_output=False)
        assert len(logger.handlers) == 1

    def test_json_format(self):
        logger = setup_logging(json_output=True)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_remove_old_handlers(self):
        logging.getLogger().addHandler(logging.StreamHandler())
        setup_logging()
        assert len(logging.getLogger().handlers) == 1


class TestStageLog:
    def test_stage_log_decorator(self, caplog):
        caplog.set_level(logging.INFO)
        logger = logging.getLogger('test_stage')

        @stage_log(logger=logger)
        def my_stage(x):
            return x * 2

        result = my_stage(21)
        assert result == 42
        assert len(caplog.records) >= 2
        stage_records = [r for r in caplog.records if hasattr(r, 'stage')]
        assert len(stage_records) >= 2

    def test_stage_log_exception(self, caplog):
        caplog.set_level(logging.ERROR)
        logger = logging.getLogger('test_stage_err')

        @stage_log(logger=logger)
        def failing_stage():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            failing_stage()
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) > 0
        assert 'boom' in error_records[-1].message

    def test_stage_log_auto_logger(self, caplog):
        caplog.set_level(logging.INFO)

        @stage_log()
        def auto_stage():
            pass

        auto_stage()
        assert len(caplog.records) >= 2
