import time
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


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