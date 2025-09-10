import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from util import read_pc_np

def compute_chamfer(points_gt, points_completed):

    # use scipy KDTree for quick distance calc
    tree = KDTree(points_completed)
    dist_A = tree.query(points_gt)[0]
    tree = KDTree(points_gt)
    dist_B = tree.query(points_completed)[0]

    return np.mean(dist_A) + np.mean(dist_B)


def compute_emd(points_gt, points_completed):

    # TODO, probably computationally infeasible

    return NotImplementedError


def eval(points_gt, points_partial, points_completed, odir=None):


    pc_completed = o3d.t.geometry.PointCloud()
    pc_completed.point.positions = o3d.core.Tensor(points_completed)

    cd_incomplete = compute_chamfer(points_gt, points_partial)
    cd_completed = compute_chamfer(points_gt, points_completed)

    print(f"Number of points in partial pc: {len(points_partial)}")
    print(f"Number of points in complete pc: {len(points_gt)}")
    print(f"Number of points in completed pc: {len(points_completed)}")

    print(f"Chamfer Distance partial to gt pc: {cd_incomplete}")
    print(f"Chamfer Distance completed to gt pc: {cd_completed}")

    if odir is not None:
        if not os.path.exists(odir):
            os.makedirs(odir)

        with open(os.path.join(odir, "eval.txt"), 'w') as f:
            f.write(f"Number of points in partial pc: {len(points_partial)}\n")
            f.write(f"Number of points in complete pc: {len(points_gt)}\n")
            f.write(f"Number of points in completed pc: {len(points_completed)}\n")

            f.write(f"Chamfer Distance partial to gt pc: {cd_incomplete}\n")
            f.write(f"Chamfer Distance completed to gt pc: {cd_completed}\n")
            
    return



def main():
    base_dir = "/Stor1/wout/OcclusionPaper/data_treepointr_test"

    # TODO: chang input args to las point clouds
    file_gt = os.path.join(base_dir, "input", "ABI_2t_1cm_SOR_6_10.txt")
    file_incomplete = os.path.join(base_dir, "input", "ABI_ground_1cm_SOR_6_10.txt")
    file_completed = os.path.join(base_dir, "output", "ABI_completion.xyz")

    points_gt = read_pc_np(file_gt, delimiter=',')
    points_incomplete = read_pc_np(file_incomplete, delimiter=',')
    points_completed = read_pc_np(file_completed)

    # TODO: where to put output
    odir=os.path.join(base_dir, "eval")

    eval(points_gt, points_incomplete, points_completed)

if __name__ == "__main__":
    main()