import os
import numpy as np
import open3d as o3d

import pyvista as pv
pv.global_theme.full_screen = True

import laspy

from visualization_o3d import batch_aabbs_to_lineset, batch_aabbs_to_mesh


def o3d_mesh_to_pyvista(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)
    faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).astype(np.int64).ravel()
    pv_mesh = pv.PolyData(vertices, faces)
    
    if o3d_mesh.has_vertex_colors():
        colors = np.asarray(o3d_mesh.vertex_colors)
        pv_mesh.point_data["Colors"] = (colors * 255).astype(np.uint8)
    
    return pv_mesh
    
def o3d_lineset_to_pyvista(o3d_lineset):
    vertices = np.asarray(o3d_lineset.points)
    lines = np.asarray(o3d_lineset.lines)

    # PyVista expects a flat array like: [2, v0, v1, 2, v0, v1, ...]
    n_lines = lines.shape[0]
    cells = np.hstack([np.full((n_lines, 1), 2), lines]).astype(np.int64).ravel()

    # Create PolyData with line cells
    pv_lines = pv.PolyData()
    pv_lines.points = vertices
    pv_lines.lines = cells
    
    return pv_lines


def test_pyvista():

    # Create a cube mesh
    cube = pv.Cube()
    
    # Initialize the plotter
    plotter = pv.Plotter()
    
    # Add the cube mesh with transparency
    plotter.add_mesh(cube, opacity=0.5, color='red')
    
    # Display the plot
    plotter.show()

def pyvista_bboxs():
    
    aabbs = []
    for i in range(100):
        min_bound = np.random.rand(3) * 10
        max_bound = min_bound + [1.0, 1.0, 1.0]
        aabbs.append(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
        
    mesh = batch_aabbs_to_mesh(aabbs)
    
    pv_mesh = o3d_mesh_to_pyvista(mesh)
    
    plotter = pv.Plotter()
    
    plotter.add_mesh(pv_mesh, opacity=0.5, color='red')
    
    plotter.show()

def pyvista_lineset():
    aabbs = []
    for i in range(100):
        min_bound = np.random.rand(3) * 10
        max_bound = min_bound + [1.0, 1.0, 1.0]
        aabbs.append(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
    
    o3d_lines = batch_aabbs_to_lineset(aabbs)
    
    pv_lines = o3d_lineset_to_pyvista(o3d_lines)

    plotter = pv.Plotter()
    plotter.add_mesh(pv_lines, color="black", line_width=3)
    plotter.show()
    
def pyvista_bboxs_with_lines():
    aabbs = []
    for i in range(100):
        min_bound = np.random.rand(3) * 10
        max_bound = min_bound + [1.0, 1.0, 1.0]
        aabbs.append(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
    
    o3d_lines = batch_aabbs_to_lineset(aabbs)
    
    pv_lines = o3d_lineset_to_pyvista(o3d_lines)
    
    mesh = batch_aabbs_to_mesh(aabbs)
    
    pv_mesh = o3d_mesh_to_pyvista(mesh)

    plotter = pv.Plotter()
    plotter.add_mesh(pv_lines, color="black", line_width=3)
    plotter.add_mesh(pv_mesh, opacity=0.4, color='red')
    plotter.show()
    

HORIZONTAL_BUFFER = 10
VERTICAL_BUFFER_BOTTOM = 5
VERTICAL_BUFFER_TOP = 20
VOX_DIM = 0.1

def occmap_vis_pyvista(occmap_file, pointcloud_file):
    # read npy
    occmap = np.load(occmap_file)
    # switch x and y
    occmap = np.swapaxes(occmap, 0, 1)
    
    las = laspy.read(pointcloud_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    min_pc = np.min(points, axis=0)
    max_pc = np.max(points, axis=0)
    
    min_pc -= [HORIZONTAL_BUFFER, HORIZONTAL_BUFFER, VERTICAL_BUFFER_BOTTOM]
    max_pc += [HORIZONTAL_BUFFER, HORIZONTAL_BUFFER, VERTICAL_BUFFER_TOP]
    
    print(min_pc)
    print(max_pc)
    
    vox_dim = 0.1
    
    print((max_pc-min_pc)/vox_dim)
    
    dims = occmap.shape
    
    print(dims)
    
    crop_min = np.floor(np.divide(dims, 2)) - 40
    crop_max = np.floor(np.divide(dims, 2)) + 30

    crop_min = np.array([0, dims[1]/2-30, 50])
    crop_max = np.array([dims[0], dims[1]/2, dims[2]-50])
    
    bboxs_occl = []
    bbox_unobserved = []
    bbox_hit = []
    
    # where the fuck do these values come from
    # TODO: redo the occlusion mapping and very clearly save the min_bound and max_bound of the voxelgrid in 3D coordinates
    offset = [+0.2,+0.1,-0.6]
    # offset = [0,0,0]
    
    
    for x in range(int(crop_min[0]), int(crop_max[0])):
        for y in range(int(crop_min[1]), int(crop_max[1])):
            for z in range(int(crop_min[2]), int(crop_max[2])):
                min_bound = min_pc + np.array([x,y,z])*vox_dim + offset
                max_bound = min_bound + vox_dim
                if occmap[x,y,z] == 3:
                    # add mesh cube
                    bboxs_occl.append(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
                elif occmap[x,y,z] == 4:
                    bbox_unobserved.append(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
                elif occmap[x,y,z] == 1:
                    bbox_hit.append(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
                    
    if len(bboxs_occl) > 0:
        mesh_occl = batch_aabbs_to_mesh(bboxs_occl)
        pv_mesh_occl = o3d_mesh_to_pyvista(mesh_occl)
    
    mesh_hit = batch_aabbs_to_mesh(bbox_hit)
    pv_mesh_hit = o3d_mesh_to_pyvista(mesh_hit)
    
    if len(bbox_unobserved) > 0:
        mesh_unobserved = batch_aabbs_to_mesh(bbox_unobserved)
        pv_mesh_unobserved = o3d_mesh_to_pyvista(mesh_unobserved)
    
    plotter = pv.Plotter()
    
    if len(bboxs_occl) > 0:
        plotter.add_mesh(pv_mesh_occl, opacity=0.2, color='red')
    plotter.add_mesh(pv_mesh_hit, opacity=0.1, color='green')
    if len(bbox_unobserved) > 0:
        plotter.add_mesh(pv_mesh_unobserved, opacity=0.2, color='blue')
        
    # add point cloud
    min_pc_crop = min_pc + crop_min*vox_dim + offset
    max_pc_crop = min_pc + crop_max*vox_dim + offset
    print(min_pc_crop)
    print(max_pc_crop)
    mask = np.all((points >= min_pc_crop) & (points <= max_pc_crop), axis=1)
    points_in_crop = points[mask]
    point_cloud = pv.PolyData(points_in_crop)
    # plotter.add_points(point_cloud, style="points", point_size=2, opacity=0.5)
    plotter.add_points(point_cloud, style="points", point_size=2)
    
    plotter.show()
    

def main():

    occmap_file = "/Stor1/wout/OcclusionPaper/data_barbara/occpy_out_old/Classification_all.npy"
    pointcloud_file = "/Stor1/wout/OcclusionPaper/data_barbara/occpy_out_old/ABI_2t_1cm_SOR_6_10.las"
    occmap_vis_pyvista(occmap_file, pointcloud_file)

if __name__ == "__main__":
    main()