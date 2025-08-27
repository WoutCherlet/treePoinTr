import os
import numpy as np
import open3d as o3d

from util import read_pc_np
from evaluate_completion import compute_chamfer


def main():
    base_dir = "/Stor1/wout/OcclusionPaper/data_treepointr_test"
    file_gt = os.path.join(base_dir, "input", "ABI_2t_1cm_SOR_6_10.txt")

    file_incomplete = os.path.join(base_dir, "input", "ABI_ground_1cm_SOR_6_10.txt")

    file_completed = os.path.join(base_dir, "output", "ABI_completion.xyz")

    points_gt = read_pc_np(file_gt, delimiter=',')
    points_incomplete = read_pc_np(file_incomplete, delimiter=',')
    points_completed = read_pc_np(file_completed)

    cd_incomplete = compute_chamfer(points_gt, points_incomplete)
    cd_completed = compute_chamfer(points_gt, points_completed)

    print(f"Chamfer Distance TLS to CLS+TLS pc: {cd_incomplete}")
    print(f"Chamfer Distance completed TLS to CLS+TLS pc: {cd_completed}")

    print(f"Number of points in TLS pc: {len(points_incomplete)}")
    print(f"Number of points in completed TLS pc: {len(points_completed)}")


    # merge completed and incomplete

    points_completed_merged = np.vstack((points_incomplete, points_completed))

    pc_completed = o3d.t.geometry.PointCloud()
    pc_completed.point.positions = o3d.core.Tensor(points_completed_merged)

    # SOR filter
    pc_completed_filtered, _ = pc_completed.remove_statistical_outliers(nb_neighbors=6, std_ratio=2)

    # then voxel downsample to 1 cm
    pc_completed_f_ds = pc_completed_filtered.voxel_down_sample(voxel_size=0.01)
    points_completed_f_ds = pc_completed_f_ds.point.positions.numpy()
    cd_completed_2 = compute_chamfer(points_gt, points_completed_f_ds)

    print(f"Chamfer Distance TLS to CLS+TLS pc: {cd_incomplete}")
    print(f"Chamfer Distance completed TLS to CLS+TLS pc: {cd_completed_2}")

    print(f"Number of points in TLS pc: {len(points_incomplete)}")
    print(f"Number of points in completed TLS pc: {len(points_completed_f_ds)}")

    


if __name__ == "__main__":
    main()