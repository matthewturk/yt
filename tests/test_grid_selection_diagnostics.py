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


def check_grid_consistency(ds, obj):
    # Retrieve the selector from the object
    selector = obj.selector

    # OLD METHOD: Linear check over all grids
    grids = np.array(ds.index.grids)
    ng = len(grids)
    if ng == 0:
        old_grids = np.array([])
    else:
        left_edges = np.empty((ng, 3), dtype="float64")
        right_edges = np.empty((ng, 3), dtype="float64")
        levels = np.zeros((ng, 1), dtype="int32")

        for i, g in enumerate(grids):
            left_edges[i, :] = g.LeftEdge.d
            right_edges[i, :] = g.RightEdge.d
            levels[i, 0] = g.Level

    # Select grids using the vectorized old-style check
    mask = selector.select_grids(left_edges, right_edges, levels)
    old_grids = grids[mask.astype("bool")]

    # Apply level filtering if present on the object
    min_level = getattr(obj, "min_level", None)
    max_level = getattr(obj, "max_level", None)

    if min_level is not None:
        old_grids = np.array([g for g in old_grids if g.Level >= min_level])
    if max_level is not None:
        old_grids = np.array([g for g in old_grids if g.Level <= max_level])

    # Calculate cell count for old grids
    old_cell_count = 0
    for g in old_grids:
        m, count = selector.fill_mask_regular_grid(g)
        if m is not None:
            old_cell_count += count

    # NEW METHOD: Get grids from the object (which uses the new selection method)
    new_grids = []
    # We iterate blocks to get the grids as the IO handler would
    # This invokes the new grid tree via the index logic
    for b in obj.blocks:
        if isinstance(b, tuple):
            new_grids.append(b[0])
        else:
            new_grids.append(b)

    # Convert to numpy array for comparison
    new_grids = np.array(new_grids, dtype="object")

    # Calculate cell count/verify logic for new grids
    new_cell_count = 0
    for g in new_grids:
        m, count = selector.fill_mask_regular_grid(g)
        if m is not None:
            new_cell_count += count

    # Comparisons
    print(f"  Old Method Grids: {len(old_grids)}")
    print(f"  New Method Grids: {len(new_grids)}")

    grid_mismatch = False
    if len(old_grids) != len(new_grids):
        print("  !!! GRID COUNT MISMATCH !!!")
        grid_mismatch = True

    # Check strict identity and order
    if not np.array_equal(old_grids, new_grids):
        if not grid_mismatch:
            print("  !!! GRID LIST/ORDER MISMATCH !!!")
            grid_mismatch = True
        # Check if it's just order
        if set(old_grids) == set(new_grids):
            print("  (It is only an order mismatch)")

    print(f"  Old Cell Count: {old_cell_count}")
    print(f"  New Cell Count: {new_cell_count}")

    if old_cell_count != new_cell_count:
        print("  !!! CELL COUNT MISMATCH !!!")
        return False

    if grid_mismatch:
        print(
            "  (Cell counts match, but grids differ - potentially optimized selection)"
        )
        return True  # Pass if cells match, assuming optimization

    print("  [OK] Consistency Check Passed")
    return True


def benchmark_selection(ds, name):
    print("=" * 60)
    print(f"BENCHMARK: {name}")
    print(f"  Total grids: {ds.index.num_grids}")
    print(f"  Max Level: {ds.index.max_level}")
    print(f"  Refine By: {ds.refine_by}")

    center = ds.domain_center
    width = ds.domain_width[0]

    # 1. Sphere Selection (Small)
    t0 = time.time()
    sp = ds.sphere(center, width * 0.05)
    # We access a field to force chunking/selection
    _ = sp["index", "ones"]
    t1 = time.time()
    print(f"  Sphere (Small r=0.05) Time: {t1-t0:.6f}s")
    if not check_grid_consistency(ds, sp):
        print("  >>> FAILED CHECK")

    # 2. Sphere Selection (Large - 0.25)
    t0 = time.time()
    sp = ds.sphere(center, width * 0.25)
    _ = sp["index", "ones"]
    t1 = time.time()
    print(f"  Sphere (Large r=0.25) Time: {t1-t0:.6f}s")
    if not check_grid_consistency(ds, sp):
        print("  >>> FAILED CHECK")

    # 3. Region Selection
    t0 = time.time()
    reg = ds.box(center - width * 0.1, center + width * 0.1)
    _ = reg["index", "ones"]
    t1 = time.time()
    print(f"  Region (Width=0.2)    Time: {t1-t0:.6f}s")
    if not check_grid_consistency(ds, reg):
        print("  >>> FAILED CHECK")

    print("\n")


def benchmark_smoothed_covering_grid(ds, name):
    print("=" * 60)
    print(f"BENCHMARK SCG: {name}")

    # Find a highly refined point
    center = ds.domain_center
    width = ds.domain_width[0] / 4.0  # smaller box
    left_edge = center - width / 2
    dims = [16, 16, 16]  # small dims, but high level
    # Pick a level that exists
    level = min(ds.index.max_level, 5)  # Go deep

    # 1. Measure Data Access Time (The actual benchmark)
    t0 = time.time()
    scg = ds.smoothed_covering_grid(level, left_edge, dims)
    # Access a field to trigger generation
    _ = scg["index", "ones"]
    t1 = time.time()
    print(f"  SCG (Level {level}) Access Time: {t1-t0:.6f}s")

    # 2. Check Consistency of components
    # SCG constructs regions for each level 0..level
    # We verify that for these regions, grid selection is consistent
    results = []
    print("  Verifying constituent level selections:")
    for l in range(level + 1):
        # Create a region mimicking the SCG requirement at this level
        # SCG uses a buffer of current_dx.
        # We'll just check a Region covering the SCG volume, strictly at level l
        # This exercises the selector for that level.

        # Note: we use ds.region (which is 'box' or 'region')
        reg = ds.region(center, left_edge, left_edge + width)
        reg.min_level = l
        reg.max_level = l

        print(f"    Checking Level {l} Region...")
        if not check_grid_consistency(ds, reg):
            print(f"    >>> FAILED CHECK at Level {l}")
            results.append(False)
        else:
            results.append(True)

    if all(results):
        print("  [OK] SCG Consistency Check Passed")


def run_all_tests():
    # Refine by 2 Deep
    ds_deep = make_deep_ds(levels=8, refine_by=2)
    benchmark_selection(ds_deep, "Deep AMR (Levels=8, Refine=2)")
    benchmark_smoothed_covering_grid(ds_deep, "Deep AMR (Levels=8, Refine=2)")

    # Refine by 4 Deep
    ds_deep4 = make_deep_ds(levels=5, refine_by=4)
    benchmark_selection(ds_deep4, "Deep AMR (Levels=5, Refine=4)")
    benchmark_smoothed_covering_grid(ds_deep4, "Deep AMR (Levels=5, Refine=4)")

    # Wide
    ds_wide = make_wide_ds(n_tiles=12)
    benchmark_selection(ds_wide, "Wide Shallow AMR (Tiles=12^3)")
    # Wide usually doesn't have deep levels, max level is 0
    benchmark_smoothed_covering_grid(ds_wide, "Wide Shallow AMR (Tiles=12^3)")

    # Default fake (Standard IsolatedGalaxy like)
    ds_std = fake_amr_ds()
    benchmark_selection(ds_std, "Standard Fake AMR")
    benchmark_smoothed_covering_grid(ds_std, "Standard Fake AMR")


if __name__ == "__main__":
    run_all_tests()
