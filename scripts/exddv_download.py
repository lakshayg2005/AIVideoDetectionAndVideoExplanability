"""Download the FaceForensics++ c23 videos used by ExDDV, retrying on network errors.

kagglehub resumes partial downloads, so retrying is cheap. Prints the dataset
path on the last line of stdout. Public dataset: no Kaggle account is needed.
"""
import sys
import time

import kagglehub

HANDLE = "xdxd003/ff-c23"
MAX_ATTEMPTS = 20

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        print(kagglehub.dataset_download(HANDLE), flush=True)
        break
    except Exception as exc:  # kagglehub raises many types for network failures
        print(f"attempt {attempt}/{MAX_ATTEMPTS} failed: {exc!r}", file=sys.stderr, flush=True)
        if attempt == MAX_ATTEMPTS:
            raise
        time.sleep(min(30 * attempt, 300))
