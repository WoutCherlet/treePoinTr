import os
import numpy as np
import open3d as o3d
import laspy

# TODO: filter here?

def merge_blocks(output_dir, inner_size, outer_size, odir):

    # TODO: get all prediction blocks, keep only points within inner box, then merge

    return

def post_process(points_partial, points_output, odir=None):

    print("Merging partial and completed points")
    points_completed_merged = np.vstack((points_partial, points_output))
    pc_completed = o3d.t.geometry.PointCloud()
    pc_completed.point.positions = o3d.core.Tensor(points_completed_merged)

    print("Applying SOR filter")
    pc_completed_filtered, _ = pc_completed.remove_statistical_outliers(nb_neighbors=6, std_ratio=2)

    print("Voxel downsampling to 1 cm")
    pc_completed_f_ds = pc_completed_filtered.voxel_down_sample(voxel_size=0.01)
    points_completed_f_ds = pc_completed_f_ds.point.positions.numpy()

    if odir is not None:
        if not os.path.exists(odir):
            os.makedirs(odir)
        o3d.t.io.write_point_cloud(os.path.join(odir, "completed.ply"), pc_completed_f_ds)

    return points_completed_f_ds

def main():
    blocks_dir = ""

    inner_size = 1.0
    outer_size = 2.0

    odir = ""

    merge_blocks(blocks_dir, inner_size, outer_size)

if __name__ == "__main__":
    main()
