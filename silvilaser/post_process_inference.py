import glob
import os
import numpy as np
import open3d as o3d

def denormalize_points(points, centroid, scale):
    return points * scale + centroid

def undo_scale_and_shift(blocks_dir, inference_dir, odir):

    os.makedirs(odir, exist_ok=True)

    for dir in glob.glob(os.path.join(inference_dir, "*")):
        if not os.path.isdir(dir):
            continue

        id, block_path = os.path.basename(dir).split('_', 1)

        npz_file = os.path.join(blocks_dir, f"{block_path}.npz")
        block_arrs = np.load(npz_file)
        
        p_completed = np.load(os.path.join(dir, 'compl_fine.npy'))
        p_compl_denormalized = denormalize_points(p_completed, block_arrs['centroid'], block_arrs['scale'])

        if not os.path.exists(os.path.join(odir, f"{id}")):
            os.makedirs(os.path.join(odir, f"{id}"))

        write_path = os.path.join(odir, f"{id}", f"{block_path}_compl_denormalized.npy")
        np.save(write_path, p_compl_denormalized)

        break        


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
    # blocks_dir = ""

    # inner_size = 1.0
    # outer_size = 2.0

    # odir = ""

    # merge_blocks(blocks_dir, inner_size, outer_size)

    blocks_dir = "/Stor1/wout/OcclusionPaper/CLS_experiment/all_cubes/PER2/blocks/"
    inference_dir = "/Stor1/wout/OcclusionPaper/CLS_experiment/inference_results/train_test_1/"
    odir = "/Stor1/wout/OcclusionPaper/CLS_experiment/processed_results/train_test_1/denormalized_blocks/"

    undo_scale_and_shift(blocks_dir, inference_dir, odir)

if __name__ == "__main__":
    main()
