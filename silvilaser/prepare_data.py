import os
import random
import numpy as np
import open3d as o3d

from util import read_pc_o3d


def get_non_occluded_pc(pc_occluded, pc_complete, distance_th = 0.01):
    '''
    Returns points in complete pointcloud that are not present in occluded point cloud

    Works by distance-based propagation of points in occluded pc
    '''
    if isinstance(pc_occluded, o3d.t.geometry.PointCloud):
        pc_occluded = pc_occluded.to_legacy()
        pc_complete = pc_complete.to_legacy()

    distances = pc_complete.compute_point_cloud_distance(pc_occluded)
    distances = np.asarray(distances)
    occluded_ind = np.where(distances < distance_th)[0]

    all_points = np.asarray(pc_complete.points)
    mask = np.ones(all_points.shape[0], dtype=bool)
    mask[occluded_ind] = False
    leftover_points = all_points[mask]

    leftover_pc = o3d.t.geometry.PointCloud()
    leftover_pc.point.positions = o3d.core.Tensor(leftover_points)

    return leftover_pc


def get_cubes(pc_occluded, pc_complete, odir, inner_size, outer_size):
    '''
        Voxelize the given point cloud and classify each voxel as suited for training/testing model

        Also saves used voxel grid for future reference
    '''

    os.makedirs(odir, exist_ok=True)

    # pc_leftover = get_non_occluded_pc(pc_occluded, pc_complete)
    # TEMP FOR DEBUG: just read
    pc_occluded = pc_occluded.to_legacy()
    pc_complete = pc_complete.to_legacy()
    pc_leftover = o3d.io.read_point_cloud("/Stor1/wout/OcclusionPaper/data_treepointr_test/input/processed/ABI_CLS_only.ply")

    min_bound = pc_leftover.get_min_bound()
    max_bound = pc_leftover.get_max_bound()

    # outer_size is entire voxel, inner_size is where predictions are kept. So we want stride to be inner_size, so we do predictions everywhere
    stride = inner_size
    diff = outer_size - inner_size

    xs = np.arange(min_bound[0]-diff, max_bound[0]-diff, stride)
    ys = np.arange(min_bound[1]-diff, max_bound[1]-diff, stride)
    zs = np.arange(min_bound[2]-diff, max_bound[2]-diff, stride)

    print(f"Total number cubes: {len(xs)} (x) * {len(ys)} (y) * {len(zs)} (z) = {len(xs)*len(ys)*len(zs)}")

    # TODO: TEMP: sample random cubes
    # for i,x in enumerate(xs):
    #     print(f"x: {i+1} / {len(xs)}")
    #     for y in ys:
    #         for z in zs:

    for i in range(1000000):
        x = random.choice(xs)
        y = random.choice(ys)
        z = random.choice(zs)
        # get bbox of current voxel
        min_b = np.array([x,y,z])
        max_b = np.array([x+outer_size,y+outer_size,z+outer_size])

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)

        inner_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b+[diff/2,diff/2,diff/2], max_bound=max_b-[diff/2,diff/2,diff/2])

        cropped_occluded = pc_occluded.crop(bbox)
        cropped_occluded_inner = cropped_occluded.crop(inner_bbox)
        if cropped_occluded.is_empty():
            continue

        cropped_leftover = pc_leftover.crop(bbox)
        cropped_leftover_inner = cropped_leftover.crop(inner_bbox)
        if cropped_leftover.is_empty() or cropped_leftover_inner.is_empty():
            continue

        if len(cropped_occluded.points) < 2000 or len(cropped_leftover_inner.points) < 1000:
            continue

        print(len(cropped_occluded.points))
        print(len(cropped_leftover_inner.points))

        # TEMP: visualize
        bbox.color = np.array([0.0, 0.0, 0.0])
        cropped_occluded.paint_uniform_color([0.0, 0.0, 1.0])
        cropped_leftover.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.visualization.draw_geometries([cropped_occluded, cropped_leftover, bbox])


    return

# def process_pointcloud(partial_points, complete_points, block_size=2.0, overlap=0.5, voxel_size=0.02, n_input=2048, n_target=8192, outdir="processed"):

#     os.makedirs(outdir, exist_ok=True)

#     blocks_partial = split_into_blocks(partial_points, block_size, overlap)
#     blocks_complete = split_into_blocks(complete_points, block_size, overlap)

#     # correspondences between blocks are on origin
#     dict_complete = {tuple(origin): pts for pts, origin in blocks_complete}

#     block_id = 0
#     for partial, origin in blocks_partial:
#         key = tuple(origin)
#         if key not in dict_complete:
#             continue
#         complete = dict_complete[key]

#         # TODO: here: check if enough points in complete and enough points in partial

#         # normalize
#         partial, centroid, scale = normalize_block(partial)
#         complete = (complete - centroid) / scale

#         # TODO: duplicate points if necessary, otherwise furthest point downsample
#         partial = farthest_point_sampling(partial, n_input)
#         complete = farthest_point_sampling(complete, n_target)

#         # Save
#         np.savez(os.path.join(outdir, f"block_{block_id:05d}.npz"),
#                  partial=partial.astype(np.float32),
#                  complete=complete.astype(np.float32),
#                  origin=origin, centroid=centroid, scale=scale)
#         block_id += 1

#     print(f"Saved {block_id} blocks to {outdir}")



def main():

    file_occl = "/Stor1/wout/OcclusionPaper/data_treepointr_test/input/ABI_ground_1cm_SOR_6_10.txt"
    file_compl = "/Stor1/wout/OcclusionPaper/data_treepointr_test/input/ABI_2t_1cm_SOR_6_10.txt"

    pc_occl = read_pc_o3d(file_occl, delimiter=",")
    pc_compl = read_pc_o3d(file_compl, delimiter=",")

    odir = "/Stor1/wout/OcclusionPaper/data_treepointr_test/ABI_processing"

    # leftover_pc = get_non_occluded_pc(pc_occl, pc_compl)

    # ofile = "/Stor1/wout/OcclusionPaper/data_treepointr_test/input/processed/ABI_CLS_only.ply"

    # o3d.t.io.write_point_cloud(ofile, leftover_pc)

    get_cubes(pc_occl, pc_compl, odir, inner_size=1.0, outer_size=2.0)


if __name__ == "__main__":
    main()