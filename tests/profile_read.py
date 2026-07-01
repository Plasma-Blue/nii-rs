"""
Profile nii-rs read path to find bottleneck.
"""

import gc
import os
import time

import numpy as np

PTH = r"E:\Dev.Repos\nii-rs\tests\spleen_11.nii.gz"
from nii._nii import Nifti1Image


# 1. Read only (decompress + parse header, no Python numpy conversion)
def bench(label, fn, n=3):
    times = []
    for _ in range(n):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / n
    print(f"  {label}: {avg * 1000:.0f} ms")


# Read + ndarray (clone)
bench("read + ndarray  ", lambda: Nifti1Image.read(PTH).ndarray())

# Read + into_ndarray (zero copy)
bench("read + into     ", lambda: Nifti1Image.read(PTH).into_ndarray())

# Read only (don't get array)
bench("read only       ", lambda: Nifti1Image.read(PTH))

# Just Python gzip decompress
import gzip

with open(PTH, "rb") as f:
    raw = f.read()
bench("python gzip dec ", lambda: gzip.decompress(raw))

# Compare with nibabel read-only
import nibabel as nib

bench("nibabel load    ", lambda: nib.load(PTH))
bench("nibabel get_fdata", lambda: nib.load(PTH).get_fdata())
