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

    points_gt = read_pc_np(file_gt)
    point_incomplete = read_pc_np(file_incomplete)
    points_completed = read_pc_np(file_completed)

    cd_incomplete = compute_chamfer(points_gt, point_incomplete)
    cd_completed = compute_chamfer(points_gt, points_completed)

    print(f"Chamfer Distance TLS to CLS+TLS pc: {cd_incomplete}")
    print(f"Chamfer Distance completed TLS to CLS+TLS pc: {cd_completed}")


if __name__ == "__main__":
    main()