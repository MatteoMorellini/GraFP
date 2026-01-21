import numpy as np
import os

def npy_to_mm(npy_path, mm_path, shape_path=None, dtype="float32"):
    """
    Convert a .npy array to a raw .mm memmap file + (optional) shape .npy.
    - Writes memmap in C-order (row-major), same as typical numpy arrays.
    """
    x = np.load(npy_path, mmap_mode="r")  # doesn't load whole array into RAM
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array (N, D). Got shape={x.shape}")

    N, D = x.shape
    os.makedirs(os.path.dirname(mm_path) or ".", exist_ok=True)

    mm = np.memmap(mm_path, dtype=dtype, mode="w+", shape=(N, D))
    mm[:] = x.astype(dtype, copy=False)   # copy into the memmap
    mm.flush()
    del mm

    if shape_path is None:
        shape_path = os.path.splitext(mm_path)[0] + "_shape.npy"
    np.save(shape_path, np.array([N, D], dtype=np.int64))

    print(f"Saved: {mm_path}")
    print(f"Saved: {shape_path} (shape={(N, D)}, dtype={dtype})")

def main():
    npy_to_mm("output/fingerprints.npy", "output/db.mm")

if __name__ == '__main__':
    main()
# Example:
# npy_to_mm("output/fingerprints.npy", "output/db.mm")
