import logging
import sys


def setup_logging(level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter('%(levelname)s | %(message)s')
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)

    return root
