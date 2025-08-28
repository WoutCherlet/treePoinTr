import os
import random
import numpy as np
import open3d as o3d

import timeit
from itertools import product
from util import read_pc_o3d, write_points_np


def get_to_complete_pc(pc_partial, pc_complete, distance_th = 0.01):
    '''
    Returns points in complete pointcloud that are not present in occluded point cloud

    Works by distance-based propagation of points in occluded pc
    '''
    if isinstance(pc_partial, o3d.t.geometry.PointCloud):
        pc_partial = pc_partial.to_legacy()
        pc_complete = pc_complete.to_legacy()

    distances = pc_complete.compute_point_cloud_distance(pc_partial)
    distances = np.asarray(distances)
    occluded_ind = np.where(distances < distance_th)[0]

    all_points = np.asarray(pc_complete.points)
    mask = np.ones(all_points.shape[0], dtype=bool)
    mask[occluded_ind] = False
    leftover_points = all_points[mask]

    leftover_pc = o3d.t.geometry.PointCloud()
    leftover_pc.point.positions = o3d.core.Tensor(leftover_points)

    return leftover_pc


def pad_points(points, N):
    l = points.shape[0]
    idx = np.random.choice(l, N - l)
    return np.concatenate([points, points[idx]], axis=0)

def farthest_point_sampling(points, N):
    l = points.shape[0]
    selected_idxs = np.zeros(N, dtype=np.int32)
    distances = np.ones(l, dtype=np.float32)*1e10
    farthest = np.random.randint(0, l)

    for i in range(N):
        selected_idxs[i] = farthest
        fp = points[farthest, :]
        dist = np.sum((points - fp) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    
    return points[selected_idxs]

def normalize_block(points):
    """
    Normalize points to unit cube

    Returns normalized points

    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale
    return points, centroid, scale

def get_cubes(pc_partial, pc_complete, inner_size, outer_size, odir, n_points_in = 2048, n_points_out = 4096):
    '''
    Voxelize the partial and complete point clouds

    Outer size is total voxel size, inner size is where predictions will be kept. So stride is inner size so predictions are made everywhere

    n_points_in and n_points_out should be constant for poinTr input and output
    '''

    # minimum amount of points that should be in complete point cloud but not in partial
    MIN_POINTS_TO_COMPLETE = 1024
    # minimum amount of points that should be in partial point cloud
    MIN_POINTS_PARTIAL = 1024

    os.makedirs(odir, exist_ok=True)
    os.makedirs(os.path.join(odir, "blocks"), exist_ok=True)
    # TODO: TEMP
    os.makedirs(os.path.join(odir, "debug"), exist_ok=True)

    # get pc of points only present in complete and not in partial
    pc_to_complete = get_to_complete_pc(pc_partial, pc_complete)

    points_complete = pc_complete.point.positions.numpy()
    points_to_complete = pc_to_complete.point.positions.numpy()
    points_partial = pc_partial.point.positions.numpy()

    min_bound = np.min(points_complete, axis=0)
    max_bound = np.max(points_complete, axis=0)

    # outer_size is entire voxel, inner_size is where predictions are kept. So we want stride to be inner_size, so we do predictions everywhere
    stride = inner_size
    diff = outer_size - inner_size

    xs = np.arange(min_bound[0]-diff, max_bound[0]-diff, stride)
    ys = np.arange(min_bound[1]-diff, max_bound[1]-diff, stride)
    zs = np.arange(min_bound[2]-diff, max_bound[2]-diff, stride)
    print(f"Total number cubes: {len(xs)} (x) * {len(ys)} (y) * {len(zs)} (z) = {len(xs)*len(ys)*len(zs)}")

    np.savez(os.path.join(odir, f"cubes_settings.npz"),
                        xs=xs,
                        ys=ys,
                        zs=zs,
                        outer_size=outer_size, 
                        inner_size=inner_size,
                        min_points_to_complete=MIN_POINTS_TO_COMPLETE,
                        min_points_partial=MIN_POINTS_PARTIAL)

    for i,x in enumerate(xs):
        print(f"x: {i+1} / {len(xs)}")
        for j,y in enumerate(ys):
            for k,z in enumerate(zs):
                # get poins in block only in complete and not in partial
                start_time = timeit.default_timer()
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
                # partial + to_complete = complete, avoids another masking operation
                block_pts_complete = np.vstack((block_pts_partial, block_pts_to_complete))
                elapsed = timeit.default_timer() - start_time
                print(f"Time to filter points: {elapsed}")

                # TODO: TEMP: save as ply to inspect
                mask_complete_orig = (
                    (points_complete[:, 0] >= x) & (points_complete[:, 0] < x + outer_size) &
                    (points_complete[:, 1] >= y) & (points_complete[:, 1] < y + outer_size) &
                    (points_complete[:, 2] >= z) & (points_complete[:, 2] < z + outer_size)
                )
                block_pts_complete_orig = points_complete[mask_complete_orig]
                write_points_np(block_pts_partial, os.path.join(odir, "debug", f"orig_block_partial_{i:03d}_{j:03d}_{k:03d}.ply"))
                write_points_np(block_pts_complete, os.path.join(odir, "debug", f"orig_block_complete_{i:03d}_{j:03d}_{k:03d}.ply"))
                write_points_np(block_pts_complete_orig, os.path.join(odir, "debug", f"orig_block_complete_orig_{i:03d}_{j:03d}_{k:03d}.ply"))

                start_time = timeit.default_timer()

                # pad or sample to get constant output points
                if len(block_pts_complete) < n_points_out:
                    block_pts_complete = pad_points(block_pts_complete, n_points_out)
                elif len(block_pts_complete) > n_points_out:
                    block_pts_complete = farthest_point_sampling(block_pts_complete, n_points_out)

                # pad or sample to get constant input points
                if len(block_pts_partial) < n_points_in:
                    block_pts_partial = pad_points(block_pts_partial, n_points_in)
                elif len(block_pts_partial) > n_points_in:
                    block_pts_partial = farthest_point_sampling(block_pts_partial, n_points_in)

                elapsed = timeit.default_timer() - start_time
                print(f"Time to pad/sample points: {elapsed}")

                # normalize
                start_time = timeit.default_timer()
                block_pts_partial, centroid, scale = normalize_block(block_pts_partial)
                block_pts_complete = (block_pts_complete - centroid) / scale
                elapsed = timeit.default_timer() - start_time
                print(f"Time to normalize points: {elapsed}")

                # Save
                start_time = timeit.default_timer()
                np.savez(os.path.join(odir, "blocks", f"block_{x:03d}_{y:03d}_{z:03d}.npz"),
                        partial=block_pts_partial.astype(np.float32),
                        complete=block_pts_complete.astype(np.float32),
                        origin=np.array([x,y,z]), centroid=centroid, scale=scale)
                
                # TODO: TEMP: save as ply to inspect
                write_points_np(block_pts_partial, os.path.join(odir, "debug", f"block_partial_{i:03d}_{j:03d}_{k:03d}.ply"))
                write_points_np(block_pts_complete, os.path.join(odir, "debug", f"block_complete_{i:03d}_{j:03d}_{k:03d}.ply"))
                elapsed = timeit.default_timer() - start_time
                print(f"Time to save points: {elapsed}")

                # TODO: TEMP: single block
                return

    return

def main():

    file_occl = "/Stor1/wout/OcclusionPaper/data_treepointr_test/input/ABI_ground_1cm_SOR_6_10.txt"
    file_compl = "/Stor1/wout/OcclusionPaper/data_treepointr_test/input/ABI_2t_1cm_SOR_6_10.txt"

    pc_occl = read_pc_o3d(file_occl, delimiter=",")
    pc_compl = read_pc_o3d(file_compl, delimiter=",")

    odir = "/Stor1/wout/OcclusionPaper/data_treepointr_test/ABI_processing"

    inner_size = 1
    outer_size = 2

    print(pc_compl.get_min_bound())
    print(pc_compl.get_max_bound())

    # get_cubes(pc_occl, pc_compl, inner_size, outer_size, odir)

    # leftover_pc = get_non_occluded_pc(pc_occl, pc_compl)

    # ofile = "/Stor1/wout/OcclusionPaper/data_treepointr_test/input/processed/ABI_CLS_only.ply"

    # o3d.t.io.write_point_cloud(ofile, leftover_pc)



if __name__ == "__main__":
    main()