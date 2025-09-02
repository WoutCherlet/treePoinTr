import os
import numpy as np
import open3d as o3d
import laspy

def read_pc_o3d(file, delimiter=" "):
    extension = file[-4:]

    if extension == ".ply":
        pc = o3d.t.io.read_point_cloud(file)
    elif extension == ".xyz" or extension == ".txt":
        # assumes space between coords
        points = np.loadtxt(file, delimiter=delimiter)

        if not points.shape[-1] == 3:
            print(f"WARNING: file {file} does not contain 3 columns, using first 3 columns")
            points = points[:,:3]
        
        pc = o3d.t.geometry.PointCloud()
        pc.point.positions = o3d.core.Tensor(points)
    elif extension == ".las":
        las = laspy.read(file)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        pc = o3d.t.geometry.PointCloud()
        pc.point.positions = o3d.core.Tensor(points)
    else:
        print(f"WARNING: extension {extension} not supported")
        os._exit(1)

    return pc

def read_pc_np(file, delimiter=" "):
    extension = file[-4:]

    if extension == ".ply":
        pc = o3d.t.io.read_point_cloud(file)
        points = pc.point.position.numpy()
    elif extension == ".xyz" or extension == ".txt":
        # assumes space between coords
        points = np.loadtxt(file, delimiter=delimiter)

        if not points.shape[-1] == 3:
            print(f"WARNING: file {file} does not contain 3 columns, using first 3 columns")
            points = points[:,:3]
    elif extension == ".las":
        las = laspy.read(file)
        points = np.vstack((las.x, las.y, las.z)).transpose()
    else:
        print(f"WARNING: extension {extension} not supported")
        os._exit(1)

    return points


def write_points_np(points, ofile):

    # TODO: option for color?

    extension = ofile[-3:]
    extensions = ["ply", "xyz", "txt"]
    if extension not in extensions:
        print(f"Warning: extensions {extensions} not in supported extensions {extensions}")

    if extension == "ply":
        pc = o3d.t.geometry.PointCloud()
        pc.point.positions = o3d.core.Tensor(points)

        o3d.t.io.write_point_cloud(ofile, pc)
    elif extension == "xyz" or extension == "txt":
        np.savetxt(ofile, points)

    return

def write_pc_o3d(pc_o3d, ofile):

    extension = ofile[-3:]
    extensions = ["ply", "xyz", "txt"]
    if extension not in extensions:
        print(f"Warning: extensions {extensions} not in supported extensions {extensions}")

    if extension == ".ply":
        if isinstance(pc_o3d, o3d.t.geometry.PointCloud):
            o3d.t.io.write_point_cloud(ofile, pc_o3d)
        elif isinstance(pc_o3d, o3d.geometry.PointCloud):
            o3d.io.write_point_cloud(ofile, pc_o3d)
    elif extension == "xyz" or extension == "txt":
        # TODO: color?
        if isinstance(pc_o3d, o3d.t.geometry.PointCloud):
            points = pc_o3d.point.positions.numpy()
        elif isinstance(pc_o3d, o3d.geometry.PointCloud):
            points = np.asarray(pc_o3d.points)

        np.savetxt(ofile, points)

    return
