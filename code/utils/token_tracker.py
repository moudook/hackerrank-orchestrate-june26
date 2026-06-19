import time


class TokenTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.start_time = time.time()

    def add_input(self, n):
        self.input_tokens += n

    def add_output(self, n):
        self.output_tokens += n
        self.calls += 1

    def get_cost(self):
        return (self.input_tokens / 1e6 * 0.10) + (self.output_tokens / 1e6 * 0.40)

    def elapsed_seconds(self):
        return time.time() - self.start_time

    def summary(self):
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_calls': self.calls,
            'estimated_cost': self.get_cost(),
            'elapsed_seconds': self.elapsed_seconds()
        }
