import os
import numpy as np
import open3d as o3d

import pyvista as pv
pv.global_theme.full_screen = True

import laspy

from visualization_o3d import batch_aabbs_to_lineset, batch_aabbs_to_mesh
from util import read_pc_np


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
    

def occmap_vis_pyvista(occmap_file, pointcloud_file, settings_file):
    # read npy
    occmap = np.load(occmap_file)
    
    # read occpy settings to align pointcloud and grid
    settings = np.load(settings_file, allow_pickle=True)
    min_occpy_grid = settings["min_bound"]
    vox_dim = settings["vox_dim"]
    h_buffer = settings["h_buffer"]
    v_buffer_bottom = settings["v_buffer_bottom"]
    v_buffer_top = settings["v_buffer_top"]
        
    las = laspy.read(pointcloud_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    min_pc = np.min(points, axis=0)
    max_pc = np.max(points, axis=0)
    
    min_pc -= [h_buffer, h_buffer, v_buffer_bottom]
    max_pc += [h_buffer, h_buffer, v_buffer_top]
    
    vox_dim = 0.1
    
    crop_min = np.asarray([200, 150, 100])
    crop_max = np.asarray([350, 300, 300])
    
    bboxs_occl = []
    bbox_unobserved = []
    bbox_hit = []
    
    for x in range(int(crop_min[0]), int(crop_max[0])):
        for y in range(int(crop_min[1]), int(crop_max[1])):
            for z in range(int(crop_min[2]), int(crop_max[2])):
                min_bound = min_occpy_grid + np.array([x,y,z])*vox_dim
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
    min_pc_crop = min_occpy_grid + crop_min*vox_dim
    max_pc_crop = min_occpy_grid + crop_max*vox_dim
    mask = np.all((points >= min_pc_crop) & (points <= max_pc_crop), axis=1)
    points_in_crop = points[mask]
    point_cloud = pv.PolyData(points_in_crop)
    # plotter.add_points(point_cloud, style="points", point_size=2, opacity=0.5)
    plotter.add_points(point_cloud, style="points", point_size=2)
    
    plotter.show()


def visualize_block_result(blocks_dir, inference_dir, block_id):
    path_original = os.path.join(blocks_dir, f"{block_id}.npz")
    path_result = os.path.join(inference_dir, f"PER2_{block_id}", "compl_fine.npy")

    block_orig = np.load(path_original)
    points_partial = block_orig['partial']
    points_complete = block_orig['complete']

    block_result = np.load(path_result)

    plotter = pv.Plotter()

    pc_partial = pv.PolyData(points_partial)
    plotter.add_points(pc_partial, style="points", point_size=5, color='blue', opacity=0.5)
    pc_complete = pv.PolyData(points_complete)
    plotter.add_points(pc_complete, style="points", point_size=5, color='green', opacity=0.5)
    pc_result = pv.PolyData(block_result)
    plotter.add_points(pc_result, style="points", point_size=5, color='red', opacity=0.5)

    plotter.show()

    return

def visualize_block_with_occmap(blocks_dir, block_id, occmap_file, occmap_settings):
    path_block = os.path.join(blocks_dir, f"{block_id}.npz")
    block = np.load(path_block)
    points_partial = block['partial']
    points_complete = block['complete']
    centroid = block['centroid']
    scale = block['scale']
    origin = block['origin']
    points_partial_denormalized = points_partial * scale + centroid
    points_complete_denormalized = points_complete * scale + centroid

    # crop occmap to extent of points
    settings = np.load(occmap_settings, allow_pickle=True)
    occmap = np.load(occmap_file)
    min_occpy_grid = settings["min_bound"]
    vox_dim = settings["vox_dim"]

    # TODO: get this from the cubes_settings
    vox_size = 2
    crop_min = np.floor((origin - min_occpy_grid) / vox_dim).astype(int)
    crop_max = np.ceil((origin + np.repeat(vox_size, 3) - min_occpy_grid) / vox_dim).astype(int)

    print(occmap.shape)
    print(crop_min)
    print(crop_max)

    bboxs_occl = []
    bbox_unobserved = []
    bbox_hit = []
    
    for x in range(int(crop_min[0]), int(crop_max[0])):
        for y in range(int(crop_min[1]), int(crop_max[1])):
            for z in range(int(crop_min[2]), int(crop_max[2])):
                min_bound = min_occpy_grid + np.array([x,y,z])*vox_dim
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

    
    pc_partial = pv.PolyData(points_partial_denormalized)
    plotter.add_points(pc_partial, style="points", point_size=5, color='blue', opacity=0.5)
    pc_complete = pv.PolyData(points_complete_denormalized)
    plotter.add_points(pc_complete, style="points", point_size=5, color='green', opacity=0.5)
    
    plotter.show()

    return

def visualize_result_with_occmap(blocks_dir, inference_dir, block_id, occmap_file, occmap_settings):

    path_block = os.path.join(blocks_dir, f"{block_id}.npz")
    block = np.load(path_block)
    block_path = os.path.join(inference_dir, f"PER2_{block_id}", "compl_fine.npy")
    points = np.load(block_path)
    centroid = block['centroid']
    scale = block['scale']
    points_denormalized = points * scale + centroid
    origin = block['origin']

    # crop occmap to extent of points
    settings = np.load(occmap_settings, allow_pickle=True)
    occmap = np.load(occmap_file)
    min_occpy_grid = settings["min_bound"]
    vox_dim = settings["vox_dim"]

    # TODO: get this from the cubes_settings
    vox_size = 2
    crop_min = np.floor((origin - min_occpy_grid) / vox_dim).astype(int)
    crop_max = np.ceil((origin + np.repeat(vox_size, 3) - min_occpy_grid) / vox_dim).astype(int)


    print(occmap.shape)
    print(crop_min)
    print(crop_max)

    bboxs_occl = []
    bbox_unobserved = []
    bbox_hit = []
    
    for x in range(int(crop_min[0]), int(crop_max[0])):
        for y in range(int(crop_min[1]), int(crop_max[1])):
            for z in range(int(crop_min[2]), int(crop_max[2])):
                min_bound = min_occpy_grid + np.array([x,y,z])*vox_dim
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

    
    pc_partial = pv.PolyData(points_denormalized)
    plotter.add_points(pc_partial, style="points", point_size=5, color='red', opacity=0.5)
    
    plotter.show()

    return

def visualize_block_and_result_with_occmap(blocks_dir, inference_dir, block_id, occmap_file, occmap_settings):
    
    path_block = os.path.join(blocks_dir, f"{block_id}.npz")
    block = np.load(path_block)
    points_partial = block['partial']
    points_complete = block['complete']
    centroid = block['centroid']
    scale = block['scale']
    origin = block['origin']
    points_partial_denormalized = points_partial * scale + centroid
    points_complete_denormalized = points_complete * scale + centroid

    block_path = os.path.join(inference_dir, f"PER2_{block_id}", "compl_fine.npy")
    points = np.load(block_path)
    points_res_denormalized = points * scale + centroid

    # crop occmap to extent of points
    settings = np.load(occmap_settings, allow_pickle=True)
    occmap = np.load(occmap_file)
    min_occpy_grid = settings["min_bound"]
    vox_dim = settings["vox_dim"]

    # TODO: get this from the cubes_settings
    vox_size = 2
    crop_min = np.floor((origin - min_occpy_grid) / vox_dim).astype(int)
    crop_max = np.ceil((origin + np.repeat(vox_size, 3) - min_occpy_grid) / vox_dim).astype(int)

    print(occmap.shape)
    print(crop_min)
    print(crop_max)

    bboxs_occl = []
    bbox_unobserved = []
    bbox_hit = []
    
    for x in range(int(crop_min[0]), int(crop_max[0])):
        for y in range(int(crop_min[1]), int(crop_max[1])):
            for z in range(int(crop_min[2]), int(crop_max[2])):
                min_bound = min_occpy_grid + np.array([x,y,z])*vox_dim
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

    
    pc_partial = pv.PolyData(points_partial_denormalized)
    plotter.add_points(pc_partial, style="points", point_size=5, color='blue', opacity=0.5)
    pc_complete = pv.PolyData(points_complete_denormalized)
    plotter.add_points(pc_complete, style="points", point_size=5, color='green', opacity=0.5)       
    pc_res = pv.PolyData(points_res_denormalized)
    plotter.add_points(pc_res, style="points", point_size=5, color='red', opacity=0.5)                

    plotter.show()

    return

def DEBUG_visualize_block_with_orig_pc(blocks_dir, block_id, pc_file):
    path_block = os.path.join(blocks_dir, f"{block_id}.npz")
    block = np.load(path_block)
    points_partial = block['partial']
    points_complete = block['complete']
    centroid = block['centroid']
    scale = block['scale']
    origin = block['origin']

    points_complete_denormalized = points_complete * scale + centroid
    points_partial_denormalized = points_partial * scale + centroid

    vox_size = 2

    points_pc = read_pc_np(pc_file)
    # filter points to voxel
    mask = np.all((points_pc >= origin) & (points_pc <= origin + np.repeat(vox_size, 3)), axis=1)
    points_pc = points_pc[mask]


    plotter = pv.Plotter()

    # pc_partial = pv.PolyData(points_partial_denormalized)
    # plotter.add_points(pc_partial, style="points", point_size=5, color='blue', opacity=0.5)
    pc_complete = pv.PolyData(points_complete_denormalized)
    plotter.add_points(pc_complete, style="points", point_size=5, color='green', opacity=0.5)
    pc_orig = pv.PolyData(points_pc)
    plotter.add_points(pc_orig, style="points", point_size=5, color='red', opacity=0.5)


    plotter.show()
    return

def main():
    # occmap_settings_file = "/Stor1/wout/OcclusionPaper/data_barbara/occpy_out_old/occpy_settings.npz"
    # occmap_file = "/Stor1/wout/OcclusionPaper/data_barbara/occpy_out_old/Classification_all.npy"
    # pointcloud_file = "/Stor1/wout/OcclusionPaper/data_barbara/occpy_out_old/ABI_2t_1cm_SOR_6_10.las"
    # occmap_vis_pyvista(occmap_file, pointcloud_file, occmap_settings_file)

    blocks_dir = "/Stor1/wout/OcclusionPaper/CLS_experiment/all_cubes/PER2/blocks/"
    inference_dir = "/Stor1/wout/OcclusionPaper/CLS_experiment/inference_results/train_test_1"
    block_id = "block_016_016_032"

    occmap_file = "/Stor1/wout/OcclusionPaper/CLS_experiment/occpy_output/PER2_nocanopy/Classification.npy"
    occmap_settings_file = "/Stor1/wout/OcclusionPaper/CLS_experiment/occpy_output/PER2_nocanopy/occpy_settings.npz"

    pc_file = "/Stor1/wout/OcclusionPaper/data_barbara/tree_pointclouds/pc_PER2_clstls_1cm_SOR_6_10.las" 

    # visualize_block_with_occmap(blocks_dir, block_id, occmap_file, occmap_settings_file)
    visualize_result_with_occmap(blocks_dir, inference_dir, block_id, occmap_file, occmap_settings_file)
    # DEBUG_visualize_block_with_orig_pc(blocks_dir, block_id, pc_file)

if __name__ == "__main__":
    main()