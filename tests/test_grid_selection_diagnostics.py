import time
import numpy as np
import yt
from yt.testing import fake_amr_ds
from yt.loaders import load_amr_grids
import sys


def generate_random_grids(ngrids, level, left_edge, right_edge, dim):
    # Generating some random grids within a box
    dx = (right_edge - left_edge) / dim
    grids = []
    for i in range(ngrids):
        # randomly place
        pos = np.random.random(3) * (right_edge - left_edge - dx * 4) + left_edge
        # align
        start_index = np.floor((pos - left_edge) / dx)
        pos = start_index * dx + left_edge

        g = {
            "left_edge": pos,
            "right_edge": pos + dx * 4,
            "level": level,
            "dimensions": np.array([4, 4, 4], dtype="int32"),
        }
        grids.append(g)
    return grids


def make_deep_ds(levels=10, refine_by=2):
    # Nested cubes centered at 0.5, 0.5, 0.5
    grids = []
    # Root grid
    grids.append(
        {
            "left_edge": np.array([0.0, 0.0, 0.0]),
            "right_edge": np.array([1.0, 1.0, 1.0]),
            "level": 0,
            "dimensions": np.array([32, 32, 32]),
        }
    )

    center = np.array([0.5, 0.5, 0.5])
    width = 1.0
    for l in range(1, levels):
        width /= refine_by
        # center it
        le = center - width / 2.0
        re = center + width / 2.0
        dims = np.array([32, 32, 32])  # Constant dimensions
        grids.append(
            {"left_edge": le, "right_edge": re, "level": l, "dimensions": dims}
        )
    return load_amr_grids(grids, [32, 32, 32], refine_by=refine_by)


def make_wide_ds(n_tiles=10):
    # Shallow but wide parameter
    grids = []
    # We will just manually construct tiled grids
    # n_tiles^3 grids
    width = 1.0 / n_tiles
    for i in range(n_tiles):
        for j in range(n_tiles):
            for k in range(n_tiles):
                le = np.array([i, j, k], dtype="float64") * width
                re = le + width
                grids.append(
                    {
                        "left_edge": le,
                        "right_edge": re,
                        "level": 0,
                        "dimensions": np.array([8, 8, 8]),
                    }
                )
    return load_amr_grids(grids, [n_tiles * 8, n_tiles * 8, n_tiles * 8])


def fclip(val, minv, maxv):
    return np.minimum(np.maximum(val, minv), maxv)


class ManualSelector:
    def __init__(self, ds):
        self.ds = ds

    def count_cells_sphere(self, center, radius):
        # Brute force check every cell in every grid
        # This is SLOW, but correct
        count = 0
        # Convert to plain floats to avoid unit issues with fclip
        if hasattr(center, "v"):
            center = center.v
        else:
            center = np.array(center)

        if hasattr(radius, "v"):
            radius = radius.v

        radius2 = radius**2

        for g in self.ds.index.grids:
            # Quick BBox check
            # Dist to bbox
            le = g.LeftEdge.d
            re = g.RightEdge.d
            dds = g.dds.d
            dims = g.ActiveDimensions

            p = fclip(center, le, re)
            d2 = ((p - center) ** 2).sum()
            if d2 > radius2:
                continue

            child_mask = g.child_mask

            # Optimization: check if fully contained
            # If grid is fully inside sphere, add all unmasked cells
            corners = np.array(
                [
                    [le[0], le[1], le[2]],
                    [re[0], re[1], re[2]],
                    [le[0], re[1], le[2]],
                    [re[0], le[1], le[2]],
                    [le[0], le[1], re[2]],
                    [re[0], re[1], re[2]],
                    [le[0], re[1], re[2]],
                    [re[0], le[1], re[2]],
                ]
            )
            max_d2 = np.max(np.sum((corners - center) ** 2, axis=1))
            if max_d2 < radius2:
                count += child_mask.sum()
                continue

            # Otherwise we have to check positions
            x, y, z = np.mgrid[0 : dims[0], 0 : dims[1], 0 : dims[2]]
            x = (x + 0.5) * dds[0] + le[0]
            y = (y + 0.5) * dds[1] + le[1]
            z = (z + 0.5) * dds[2] + le[2]

            dist2 = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
            mask = dist2 <= radius2

            # Mask out children
            mask = mask & (child_mask == 1)

            count += mask.sum()
        return count


def benchmark_selection(ds, name):
    print("=" * 60)
    print(f"BENCHMARK: {name}")
    print(f"  Total grids: {ds.index.num_grids}")
    print(f"  Max Level: {ds.index.max_level}")
    print(f"  Refine By: {ds.refine_by}")

    # Selectors
    center = ds.domain_center
    width = ds.domain_width[0]

    # 1. Sphere Selection (Small)
    t0 = time.time()
    sp = ds.sphere(center, width * 0.05)
    # Force count used by fast index (count is cached)
    # We access a field to force chunking if needed, but 'index' fields are fastest
    count = sp["index", "ones"].sum()
    t1 = time.time()
    print(f"  Sphere (Small r=0.05): {int(count):12d} cells  Time: {t1-t0:.6f}s")

    # 2. Sphere Selection (Large - 0.25)
    t0 = time.time()
    sp = ds.sphere(center, width * 0.25)
    count_large = sp["index", "ones"].sum()
    t1 = time.time()
    print(f"  Sphere (Large r=0.25): {int(count_large):12d} cells  Time: {t1-t0:.6f}s")

    # 3. Region Selection
    t0 = time.time()
    reg = ds.box(center - width * 0.1, center + width * 0.1)
    count_reg = reg["index", "ones"].sum()
    t1 = time.time()
    print(f"  Region (Width=0.2):    {int(count_reg):12d} cells  Time: {t1-t0:.6f}s")

    # Correctness check (Enable for small/medium datasets only, it's slow!)
    if ds.index.num_grids < 5000:
        print("  [Verifying correctness against ManualSelector...]")
        ms = ManualSelector(ds)

        # Check Small Sphere
        m_count = ms.count_cells_sphere(center, width * 0.05)
        if m_count != count:
            print(
                f"  !!! MISMATCH Small Sphere !!! Manual: {m_count}, Fast: {count}, Diff: {count - m_count}"
            )
        else:
            print("  [OK] Small Sphere MATCH")

        # Check Large Sphere
        m_count = ms.count_cells_sphere(center, width * 0.25)
        if m_count != count_large:
            print(
                f"  !!! MISMATCH Large Sphere !!! Manual: {m_count}, Fast: {count_large}, Diff: {count_large - m_count}"
            )
        else:
            print("  [OK] Large Sphere MATCH")
    print("\n")


def run_all_tests():
    # Refine by 2 Deep
    ds_deep = make_deep_ds(levels=8, refine_by=2)
    benchmark_selection(ds_deep, "Deep AMR (Levels=8, Refine=2)")

    # Refine by 4 Deep
    ds_deep4 = make_deep_ds(levels=5, refine_by=4)
    benchmark_selection(ds_deep4, "Deep AMR (Levels=5, Refine=4)")

    # Wide
    ds_wide = make_wide_ds(n_tiles=12)
    benchmark_selection(ds_wide, "Wide Shallow AMR (Tiles=12^3)")

    # Default fake (Standard IsolatedGalaxy like)
    ds_std = fake_amr_ds()
    benchmark_selection(ds_std, "Standard Fake AMR")


if __name__ == "__main__":
    run_all_tests()
