
import numpy as np
import yt
from yt.frontends.stream.data_structures import StreamHandler, StreamDataset, StreamDictFieldHandler
import time

def setup_ds(ngrids_target=10000, max_level=6):
    np.random.seed(0x4d3d3d3d)
    
    # fixed parameters
    refine_by = 2
    g_dim = 16 # 16^3 cells per grid
    
    # Level 0: 16x16x16 decomposition of domain [0,1]
    # 4096 grids initially
    n_root = 16
    
    grids = []
    
    # Level 0
    dx = 1.0 / n_root
    
    # Vectorized creation of level 0
    indices = np.indices((n_root, n_root, n_root)).reshape(3, -1).T
    les = indices * dx
    levels = np.zeros(len(les), dtype='int32')
    
    # Global storage
    all_les = [les]
    all_levels = [levels]
    all_parents = [np.full(len(les), -1, dtype='int64')]
    
    # Indices tracking
    current_les = les
    # We need to know the global index of the current level grids to assign as parents
    # The global index of the START of the current level batch
    current_global_start_idx = 0
    
    count = len(les)
    
    for lvl in range(max_level):
        if count >= ngrids_target: break
        
        current_dx = dx / (refine_by**lvl)
        
        centers = current_les + (0.5 * current_dx)
        dists = np.linalg.norm(centers - 0.5, axis=1)
        
        n_needed = (ngrids_target - count) / 8
        n_available = len(current_les)
        
        n_refine = min(int(n_needed), n_available)
        if n_refine == 0: break

        sort_inds = np.argsort(dists)
        refine_inds = sort_inds[:n_refine]
        
        # Create children
        parent_les = current_les[refine_inds]
        # Calculate GLOBAL indices of parents
        parent_global_indices = current_global_start_idx + refine_inds
        
        new_dx = current_dx / refine_by
        
        offsets = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).reshape(3, 8).T * new_dx
        
        child_les = (parent_les[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        child_levels = np.full(len(child_les), lvl + 1, dtype='int32')
        
        # Parent IDs for children: each parent has 8 children
        # parent indices repeated 8 times
        child_parents = np.repeat(parent_global_indices, 8)
        
        all_les.append(child_les)
        all_levels.append(child_levels)
        all_parents.append(child_parents)
        
        # Update for next iteration
        current_les = child_les
        current_global_start_idx = count
        count += len(child_les)
        
    print(f"Generated {count} grids.")
    print("Initializing StreamDataset...")
    
    # Flatten arrays
    tot_les = np.vstack(all_les)
    tot_levels = np.concatenate(all_levels)
    tot_parents = np.concatenate(all_parents)
    
    # Calculate Right Edges
    dx0 = 1.0 / n_root
    dxs = dx0 / (np.power(refine_by, tot_levels).astype(float))
    tot_res = tot_les + dxs[:, None]
    
    tot_dims = np.zeros((count, 3), dtype='int32')
    tot_dims[:] = g_dim
    
    # Grid levels must be (N, 1) usually? yt expects specific shapes
    grid_levels = tot_levels.reshape(count, 1).astype('int32')
    grid_left_edges = tot_les.astype('float64')
    grid_right_edges = tot_res.astype('float64')
    grid_dimensions = tot_dims.astype('int32')
    parent_ids = tot_parents.astype('int64')
    
    # Dummy particles
    number_of_particles = np.zeros((count, 1), dtype='int64')
    
    # Stream Handler setup
    sfh = StreamDictFieldHandler()
    
    domain_dimensions = np.array([n_root*g_dim]*3, dtype='int32')
    
    # We need to construct the handler manually
    handler = StreamHandler(
        grid_left_edges,
        grid_right_edges,
        grid_dimensions,
        grid_levels,
        parent_ids,
        number_of_particles,
        np.zeros(count).reshape((count, 1)), # processor_ids
        sfh,
        {}, # field_units
        ("code_length", "code_mass", "code_time", "code_velocity", "code_magnetic"),
        particle_types={},
        periodicity=(True, True, True),
        parameters={}
    )
    
    handler.name = "BenchAMR"
    handler.domain_left_edge = np.zeros(3)
    handler.domain_right_edge = np.ones(3)
    handler.refine_by = refine_by
    handler.dimensionality = 3
    handler.domain_dimensions = domain_dimensions
    handler.simulation_time = 0.0
    handler.cosmology_simulation = 0
    
    sds = StreamDataset(
        handler,
        geometry="cartesian",
        unit_system="cgs"
    )
    
    return sds

def benchmark_ghost_zones(ds, n_samples=1000):
    np.random.seed(90210)
    print("Building grid index (this may take a few minutes)...")
    t_idx_start = time.time()
    grids = ds.index.grids
    t_idx_end = time.time()
    print(f"Index build time: {t_idx_end - t_idx_start:.2f}s")
    
    indices = np.random.choice(len(grids), n_samples, replace=False)
    sample_grids = grids[indices]
    
    print(f"Benchmarking ghost zone generation for {n_samples} grids...")
    t_start = time.time()
    
    for g in sample_grids:
        level = g.Level
        g_dx = g.dds[0]
        le = g.LeftEdge
        dims = g.ActiveDimensions
        
        # Grid + 2 ghost zones
        scg_dims = dims + 4 
        scg_le = le - 2*g_dx
        
        scg = ds.smoothed_covering_grid(level, scg_le, scg_dims)
        
        # Trigger generation
        _ = scg["index", "ones"]
        
    t_end = time.time()
    print(f"Total time: {t_end - t_start:.4f} s")
    print(f"Avg time per grid: {(t_end - t_start)/n_samples:.4f} s")

if __name__ == "__main__":
    t0 = time.time()
    # Use 30k grids
    ds = setup_ds(30000, 6)
    t1 = time.time()
    print(f"Setup time: {t1-t0:.2f}s")
    benchmark_ghost_zones(ds)
