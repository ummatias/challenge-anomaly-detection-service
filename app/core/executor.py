"""
Shared ThreadPoolExecutor for fit() calls.
"""

import os
from concurrent.futures import ThreadPoolExecutor

_WORKERS = int(os.getenv("FIT_WORKERS", min(32, (os.cpu_count() or 1) + 4)))

executor = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="fit_worker")
