import os
import numpy as np
import timeit
from itertools import product

from prepare_data import get_to_complete_pc

def assign_blocks(points_partial, points_complete, inner_size, outer_size):
    N_partial = points_partial.shape[0]
    N_complete = points_complete.shape[0]
    stride = inner_size
    diff = outer_size - inner_size

    xmin, ymin, zmin = points_complete.min(axis=0) - [diff, diff, diff]
    xmax, ymax, zmax = points_complete.max(axis=0) - [diff, diff, diff]

    # Normalize coordinates so xmin,ymin,zmin -> 0
    coords_complete = points_complete - np.array([xmin, ymin, zmin])
    coords_partial = points_partial - np.array([xmin, ymin, zmin])

    # Precompute block grid extents
    nx = int(np.ceil((xmax - xmin - outer_size) / stride)) + 1
    ny = int(np.ceil((ymax - ymin - outer_size) / stride)) + 1
    nz = int(np.ceil((zmax - zmin - outer_size) / stride)) + 1

    block_map_complete = {}

    for i in range(N_complete):
        x, y, z = coords_complete[i]

        # Ranges of valid k for each axis
        kx_min = max(0, int(np.floor((x - outer_size)) / stride) + 1)
        kx_max = min(nx-1, int(np.floor(x / stride)))
        ky_min = max(0, int(np.floor((y - outer_size) / stride)) + 1)
        ky_max = min(ny-1, int(np.floor(y / stride)))
        kz_min = max(0, int(np.floor((z - outer_size) / stride)) + 1)
        kz_max = min(nz-1, int(np.floor(z / stride)))

        # Cartesian product of valid block indices
        for kx, ky, kz in product(range(kx_min, kx_max+1),
                                  range(ky_min, ky_max+1),
                                  range(kz_min, kz_max+1)):
            block_map_complete.setdefault((kx, ky, kz), []).append(i)

    block_map_partial = {}
    for i in range(N_partial):
        x, y, z = coords_partial[i]

        # Ranges of valid k for each axis
        kx_min = max(0, int(np.floor((x - outer_size)) / stride) + 1)
        kx_max = min(nx-1, int(np.floor(x / stride)))
        ky_min = max(0, int(np.floor((y - outer_size) / stride)) + 1)
        ky_max = min(ny-1, int(np.floor(y / stride)))
        kz_min = max(0, int(np.floor((z - outer_size) / stride)) + 1)
        kz_max = min(nz-1, int(np.floor(z / stride)))

        # Cartesian product of valid block indices
        for kx, ky, kz in product(range(kx_min, kx_max+1),
                                  range(ky_min, ky_max+1),
                                  range(kz_min, kz_max+1)):
            block_map_partial.setdefault((kx, ky, kz), []).append(i)


    return block_map_partial, block_map_complete

def benchmark_selection(pc_partial, pc_complete, inner_size, outer_size):
    pc_to_complete = get_to_complete_pc(pc_partial, pc_complete)

    points_complete = pc_complete.point.positions.numpy()
    points_to_complete = pc_to_complete.point.positions.numpy()
    points_partial = pc_partial.point.positions.numpy()
    
    min_bound = np.min(points_complete, axis=0)
    max_bound = np.max(points_complete, axis=0)

    stride = inner_size
    diff = outer_size - inner_size

    xs = np.arange(min_bound[0]-diff, max_bound[0]-diff, stride)
    ys = np.arange(min_bound[1]-diff, max_bound[1]-diff, stride)
    zs = np.arange(min_bound[2]-diff, max_bound[2]-diff, stride)

    # minimum amount of points that should be in complete point cloud but not in partial
    MIN_POINTS_TO_COMPLETE = 1024
    # minimum amount of points that should be in partial point cloud
    MIN_POINTS_PARTIAL = 1024
    # number of blocks to check
    N_BENCHMARK = 1000

    start_time = timeit.default_timer()
    n_selected_numpy = 0
    n_checked = 0
    for i,x in enumerate(xs):
        if n_checked > N_BENCHMARK:
            break
        for j,y in enumerate(ys):
            for k,z in enumerate(zs):
                n_checked += 1
                mask_to_complete = (
                    (points_to_complete[:, 0] >= x) & (points_to_complete[:, 0] < x + outer_size) &
                    (points_to_complete[:, 1] >= y) & (points_to_complete[:, 1] < y + outer_size) &
                    (points_to_complete[:, 2] >= z) & (points_to_complete[:, 2] < z + outer_size)
                )
                block_pts_to_complete = points_to_complete[mask_to_complete]
                if len(block_pts_to_complete) < MIN_POINTS_TO_COMPLETE:
                    continue
                # get points in block in partial
                mask_partial = (
                    (points_partial[:, 0] >= x) & (points_partial[:, 0] < x + outer_size) &
                    (points_partial[:, 1] >= y) & (points_partial[:, 1] < y + outer_size) &
                    (points_partial[:, 2] >= z) & (points_partial[:, 2] < z + outer_size)
                )
                block_pts_partial = points_partial[mask_partial]
                if len(block_pts_partial) < MIN_POINTS_PARTIAL:
                    continue

                n_selected_numpy += 1

    elapsed = timeit.default_timer() - start_time
    print(f"Time for numpy check: {elapsed} ({N_BENCHMARK} blocks)")
    print(f"Number of blocks that passed check: {n_selected_numpy}")

    start_time = timeit.default_timer()
    n_selected_o3d = 0
    n_checked = 0
    for i,x in enumerate(xs):
        if n_checked > N_BENCHMARK:
            break
        for j,y in enumerate(ys):
            for k,z in enumerate(zs):
                n_checked += 1
                min_b = np.array([x,y,z])
                max_b = np.array([x+outer_size,y+outer_size,z+outer_size])

                bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)
                block_to_complete_pc = pc_to_complete.crop(bbox)
                if block_to_complete_pc.point.positions.shape[0] < MIN_POINTS_TO_COMPLETE:
                    continue

                block_partial_pc = pc_partial.crop(bbox)
                if block_partial_pc.point.positions.shape[0] < MIN_POINTS_PARTIAL:
                    continue

                n_selected_o3d += 1

    elapsed = timeit.default_timer() - start_time
    print(f"Time for open3d check: {elapsed} ({N_BENCHMARK} blocks)")
    print(f"Number of blocks that passed check: {n_selected_o3d}")

    # IDEA: first calculate all voxel indices per point, then go through those indices
    start_time = timeit.default_timer()
    block_map_partial, block_map_to_complete = assign_blocks(points_partial, points_to_complete, inner_size, outer_size)
    elapsed = timeit.default_timer() - start_time
    print(f"Time for building assignment dictionaries: {elapsed} ({N_BENCHMARK} blocks)")
    start_time = timeit.default_timer()
    n_selected_idxing = 0
    n_checked = 0
    for vox_idx in block_map_partial:
        if n_checked > N_BENCHMARK:
            break
        n_checked += 1

        if vox_idx not in block_map_to_complete or len(block_map_to_complete[vox_idx]) < MIN_POINTS_TO_COMPLETE:
            continue

        if len(block_map_partial[vox_idx]) < MIN_POINTS_PARTIAL:
            continue

        n_selected_idxing += 1

    elapsed = timeit.default_timer() - start_time
    print(f"Time for custom check with prior assignment: {elapsed} ({N_BENCHMARK} blocks)")
    print(f"Number of blocks that passed check: {n_selected_idxing}")

    return

