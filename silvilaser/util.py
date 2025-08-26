import os
import numpy as np
import open3d as o3d

def read_pc_o3d(file):
    extension = file[-4:]

    if extension == ".ply":
        pc = o3d.t.io.read_point_cloud(file)
    elif extension == ".xyz" or extension == ".txt":
        # assumes space between coords
        points = np.loadtxt(file)

        if not points.shape[-1] == 3:
            print(f"WARNING: file {file} does not contain 3 columns")
            os._exit(1)
        
        pc = o3d.t.geometry.PointCloud()
        pc.point.positions = o3d.core.Tensor(points)
    else:
        print(f"WARNING: extension {extension} not supported")
        os._exit(1)

    return pc


def read_pc_np(file):
    extension = file[-4:]

    if extension == ".ply":
        pc = o3d.t.io.read_point_cloud(file)
        points = pc.point.position.numpy()
    elif extension == ".xyz" or extension == ".txt":
        # assumes space between coords
        points = np.loadtxt(file)

        if not points.shape[-1] == 3:
            print(f"WARNING: file {file} does not contain 3 columns")
            os._exit(1)
    else:
        print(f"WARNING: extension {extension} not supported")
        os._exit(1)

    return points
