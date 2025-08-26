import os
import numpy as np
import open3d as o3d



def get_seperate_pcs(pc_occluded, pc_complete):
    # get point in pc_complete that are not present in pc_occluded

    # TODO: propagate point of pc_occluded to pc_complete

    return 


def get_cubes(pc_occluded, pc_complete, odir, vox_dim):
    '''
        Voxelize the given point cloud and classify each voxel as suited for training/testing model

        Also saves used voxel grid for future reference
    '''

    # get min bound of pc_complete

    # loop over all voxels
        # check if empty
        # if not, check if fit for completion TODO: how? don't overthink this!
        # write to directory

    return


