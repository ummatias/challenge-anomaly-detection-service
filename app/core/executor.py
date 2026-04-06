"""
Shared ThreadPoolExecutor for fit() calls.

Why ThreadPoolExecutor and not ProcessPoolExecutor:

  - numpy releases the GIL when running operations implemented in C,
    so others threads are free to run python code
  - Processes would require pickling model state across boundaries
    and make the asyncio.Lock not work as intended ( each procces
    would have its own lock)
  - IPC cost of ProcessPoolExecutor exceeds the GIL contention

"""

import os
from concurrent.futures import ThreadPoolExecutor

_WORKERS = int(os.getenv("FIT_WORKERS", min(32, (os.cpu_count() or 1) + 4)))

executor = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="fit_worker")
