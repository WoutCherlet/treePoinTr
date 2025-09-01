import os
import numpy as np
import open3d as o3d


def visualize_cube(cube_path):

    # TODO: read .npz block and visualize


    return

def occmap_to_voxelgrid(occlusion_map):

    # TODO: occlusion map to open3d voxelgrid

    # for each occluded voxel, create an open3d mesh cube with

    return

def visualize_cube_with_occmap(cube_path, occlusion_map):

    # TODO: read .npz block and .npy occlusion map

    # visualize voxels with translucent cube? might be hard

    return


def aabb_to_mesh(aabb):
    min_x, min_y, min_z = aabb.get_min_bound()
    max_x, max_y, max_z = aabb.get_max_bound()

    # vertices
    V = np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z],
    ], dtype=np.float64)

    F = np.array([
        [0, 1, 2], [0, 2, 3],       # bottom (z=min)
        [4, 5, 6], [4, 6, 7],       # top (z=max)
        [0, 1, 5], [0, 5, 4],       # y=min face
        [3, 2, 6], [3, 6, 7],       # y=max face
        [0, 3, 7], [0, 7, 4],       # x=min face
        [1, 2, 6], [1, 6, 5],       # x=max face
    ], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    mesh.compute_vertex_normals()
    return mesh

def batch_aabbs_to_mesh(aabbs: list[o3d.geometry.AxisAlignedBoundingBox]):
    all_vertices = []
    all_triangles = []
    offset = 0

    F = np.array([
        [0, 1, 2], [0, 2, 3],       # bottom (z=min)
        [4, 5, 6], [4, 6, 7],       # top (z=max)
        [0, 1, 5], [0, 5, 4],       # y=min face
        [3, 2, 6], [3, 6, 7],       # y=max face
        [0, 3, 7], [0, 7, 4],       # x=min face
        [1, 2, 6], [1, 6, 5],       # x=max face
    ], dtype=np.int32)

    for aabb in aabbs:
        min_x, min_y, min_z = aabb.get_min_bound()
        max_x, max_y, max_z = aabb.get_max_bound()

        # vertices
        V = np.array([
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ], dtype=np.float64)

        all_vertices.append(V)
        all_triangles.append(F + offset)
        offset += V.shape[0]

    all_vertices = np.vstack(all_vertices)
    all_triangles = np.vstack(all_triangles)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(all_triangles)

    return mesh

def aabb_to_lineset(aabb):
    corners = np.asarray(aabb.get_box_points())

    # get lines
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ])

    colors = np.tile([1.0, 1.0, 1.0], (lines.shape[0], 1))

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    return lineset

def batch_aabbs_to_lineset(aabbs: list[o3d.geometry.AxisAlignedBoundingBox]):
    """
    Create a single LineSet containing many AABB wireframes.

    Parameters
    ----------
    aabbs : list of o3d.geometry.AxisAlignedBoundingBox
        List of bounding boxes.

    Returns
    -------
    lineset : o3d.geometry.LineSet
        Combined LineSet for all boxes.
    """
    all_points = []
    all_lines = []
    all_colors = []
    offset = 0

    # Each cube has the same edge structure
    cube_edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ])

    for aabb in aabbs:
        corners = np.asarray(aabb.get_box_points())
        all_points.append(corners)
        all_lines.append(cube_edges + offset)
        offset += corners.shape[0]

        # give all edges same color (white), could also be random
        all_colors.append(np.tile([1, 1, 1], (cube_edges.shape[0], 1)))

    # Stack everything
    all_points = np.vstack(all_points)
    all_lines = np.vstack(all_lines)
    all_colors = np.vstack(all_colors)

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(all_points)
    lineset.lines = o3d.utility.Vector2iVector(all_lines)
    lineset.colors = o3d.utility.Vector3dVector(all_colors)

    return lineset


def setup_camera_for_aabbs(renderer: o3d.visualization.rendering.OffscreenRenderer,
                           aabbs: list[o3d.geometry.AxisAlignedBoundingBox],
                           fov: float = 60.0,
                           distance_factor: float = 1.0):
    if len(aabbs) == 0:
        return
    
    # Combine into one bounding box that encloses all AABBs
    big_aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(
            np.vstack([np.asarray(bb.get_box_points()) for bb in aabbs])
        )
    )

    center = big_aabb.get_center()
    extent = np.linalg.norm(big_aabb.get_extent())

    # eye: sideways diagonal view
    eye = center + distance_factor * extent * np.array([1.0, 1.0, 1.0])
    # z-axis up
    up = np.array([0.0, 0.0, 1.0])
    # Apply camera
    renderer.setup_camera(fov, center, eye, up)

    return renderer


def render_offscreen_test():
    aabbs = []
    for i in range(100):
        min_bound = np.random.rand(3) * 10
        max_bound = min_bound + [1.0, 1.0, 1.0]
        aabbs.append(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))

    mesh = batch_aabbs_to_mesh(aabbs)
    mesh.compute_vertex_normals()

    # Create an offscreen renderer
    render = o3d.visualization.rendering.OffscreenRenderer(800, 600)
    render.scene.set_background([1.0, 1.0, 1.0, 0.5])

    # TODO: get 
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = [1.0, 0.0, 0.0, 0.5]

    render.scene.add_geometry("mesh", mesh, mat)

    # Correct usage of setup_camera in 0.19
    render = setup_camera_for_aabbs(render, aabbs, distance_factor=0.5)   # 60° FOV

    img = render.render_to_image()

    o3d.io.write_image("output.png", img)

def render_onscreen_test():
    # onscreen app
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Cube Viewer", 1024, 768)     
    scene_widget = o3d.visualization.gui.SceneWidget()
    scene_widget.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(scene_widget)
    scene_widget.scene.set_background([1, 1, 1, 1])

    aabbs = []
    for i in range(100):
        min_bound = np.random.rand(3) * 10
        max_bound = min_bound + [1.0, 1.0, 1.0]
        aabbs.append(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))

    mesh = batch_aabbs_to_mesh(aabbs)
    mesh.compute_vertex_normals()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = [1.0, 0.0, 0.0, 0.5]

    scene_widget.scene.add_geometry("boxes", mesh, mat)

    # setup camera
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(np.vstack([g.get_axis_aligned_bounding_box().get_box_points()
                                                for g in aabbs]))
    )
    # distance_factor = 1.0
    # extent = np.linalg.norm(bbox.get_extent())
    # eye = bbox.get_center() + distance_factor * extent * np.array([1.0, 1.0, 1.0])
    # scene.camera.look_at(bbox.get_center(), eye, np.array([0, 0, 1]))
    scene_widget.setup_camera(60, bbox, bbox.get_center())

    app.run()

def render_onscreen_test_manual():

    aabbs = []
    for i in range(100):
        min_bound = np.random.rand(3) * 10
        max_bound = min_bound + [1.0, 1.0, 1.0]
        aabbs.append(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))

    mesh = batch_aabbs_to_mesh(aabbs)
    mesh.compute_vertex_normals()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = [1.0, 0.0, 0.0, 0.5]

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(mesh)

    vis.run()
    vis.destroy_window()

    return


def main():

    # mesh = o3d.geometry.TriangleMesh.create_box()
    # mesh.compute_vertex_normals()

    render_onscreen_test()


if __name__ == "__main__":
    main()