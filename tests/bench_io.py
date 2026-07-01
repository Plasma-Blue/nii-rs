"""
I/O benchmark: nii-rs vs nibabel vs SimpleITK.

Reads a real .nii.gz file, measures time for:
  1. Read (load + decompress)
  2. Get array (ndarray access)
  3. Read + Array combined
  4. Write (array + compress)

Usage:
    python tests/bench_io.py
"""

import gc
import os
import sys
import tempfile
import time

import numpy as np

PTH = os.path.join(os.path.dirname(__file__), "spleen_11.nii.gz")
TMP = tempfile.gettempdir()
N_WARMUP = 3
N_RUNS = 5


def fmt(v: float) -> str:
    if v < 1.0:
        return f"{v * 1000:.1f} ms"
    return f"{v:.2f} s"


def bench(label: str, fn, n_runs=N_RUNS):
    # warmup
    for _ in range(N_WARMUP):
        gc.collect()
        fn()
    # measure
    times = []
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  {label}: {fmt(avg)}  (min={fmt(min(times))}, max={fmt(max(times))})")


def main():
    print(f"File: {PTH}")
    sz = os.path.getsize(PTH)
    print(f"Size: {sz / 1e6:.1f} MB (gzip)\n")

    # ─── nibabel ─────────────────────────────────────────────────────────
    print("[nibabel]")
    import nibabel as nib

    # Read (load + decompress)
    def read_nib():
        return nib.load(PTH)

    bench("load     ", read_nib)

    img_nib = read_nib()

    def getdata_nib():
        return img_nib.get_fdata()

    bench("get_fdata", getdata_nib)

    def read_all_nib():
        im = nib.load(PTH)
        _ = im.get_fdata()

    bench("load+data", read_all_nib)

    # Write
    def write_nib():
        nib.save(img_nib, os.path.join(TMP, "_bench_nib.nii.gz"))

    bench("write    ", write_nib)

    # ─── SimpleITK ───────────────────────────────────────────────────────
    print("\n[SimpleITK]")
    import SimpleITK as sitk

    def read_sitk():
        return sitk.ReadImage(PTH)

    bench("ReadImage", read_sitk)

    img_sitk = read_sitk()

    def array_sitk():
        return sitk.GetArrayFromImage(img_sitk)

    bench("GetArray  ", array_sitk)

    def read_all_sitk():
        im = sitk.ReadImage(PTH)
        _ = sitk.GetArrayFromImage(im)

    bench("read+arr  ", read_all_sitk)

    def write_sitk():
        sitk.WriteImage(img_sitk, os.path.join(TMP, "_bench_sitk.nii.gz"))

    bench("WriteImage", write_sitk)

    # ─── nii-rs ──────────────────────────────────────────────────────────
    print("\n[nii-rs  ]")
    from nii._nii import Nifti1Image as Nii

    def read_nii():
        return Nii.read(PTH)

    bench("read     ", read_nii)

    im_nii = read_nii()

    def ndarray_nii():
        return im_nii.ndarray()

    bench("ndarray  ", ndarray_nii)

    def read_all_nii():
        im = Nii.read(PTH)
        _ = im.ndarray()

    bench("read+arr ", read_all_nii)

    def write_nii():
        im_nii.write(os.path.join(TMP, "_bench_nii.nii.gz"))

    bench("write    ", write_nii)

    # ─── nii-rs into_ndarray (zero-copy path) ────────────────────────────
    print("\n[nii-rs  ] into_ndarray (zero-copy get)")

    def read_and_into():
        im = Nii.read(PTH)
        _ = im.into_ndarray()

    bench("read+into", read_and_into)

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n─── Summary ─────────────────────────────────")
    print("Best viewed as: read+arr combined time")
    print("nibabel  read+data     : run bench above")
    print("SimpleITK read+arr     : run bench above")
    print("nii-rs   read+arr      : run bench above")
    print("nii-rs   read+into     : run bench above")
    print(f"({N_WARMUP} warmup, {N_RUNS} measured runs per test)")


if __name__ == "__main__":
    main()
