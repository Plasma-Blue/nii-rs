"""
Profile nii-rs write path to find bottleneck.
"""

import gc
import os
import tempfile
import time

import numpy as np

PTH = r"E:\Dev.Repos\nii-rs\tests\spleen_11.nii.gz"
TMP = tempfile.gettempdir()
from nii._nii import Nifti1Image

im = Nifti1Image.read(PTH)
data = im.ndarray()
print(f"Shape: {data.shape}, dtype: {data.dtype}, size: {data.nbytes / 1e6:.0f} MB")


def bench(label, fn, n=3):
    gc.collect()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / n
    print(f"  {label}: {avg * 1000:.0f} ms")


# 1. Write without gzip (.nii)
pth_nii = os.path.join(TMP, "_test.nii")
bench("write .nii        ", lambda: im.write(pth_nii))

# 2. Write with gzip (.nii.gz)
pth_gz = os.path.join(TMP, "_test.nii.gz")
bench("write .nii.gz     ", lambda: im.write(pth_gz))

# 3. Just gzip the raw bytes (Python)
import gzip

with open(pth_nii, "rb") as f:
    raw_nii = f.read()
nii_mb = len(raw_nii) / 1e6
print(f"  raw .nii size: {nii_mb:.0f} MB")
bench("python gzip lvl6  ", lambda: gzip.compress(raw_nii, compresslevel=6))
bench("python gzip lvl1  ", lambda: gzip.compress(raw_nii, compresslevel=1))
