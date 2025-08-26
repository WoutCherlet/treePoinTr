

import numpy as np
import os



# function for filtering points by bounding box
def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    """ Compute a bounding_box filter on the given points
    from: https://stackoverflow.com/questions/42352622/finding-points-within-a-bounding-box-with-numpy    
    Parameters
    ----------                        
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keeped or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter


# function for cutting point cloud into cubes (voxelizing), with four different voxel sizes and center points

def cut_point_cloud(point_cloud, outpath, size1, size2, size3, size4):
    
    # Create a directory to save the .xyz files if it doesn't exist
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    # Find the minimum and maximum coordinates
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])
    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])
    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    print(f"Cutting with size 1 ({size1})")

    # Calculate the number of cubes in each dimension
    cube_size = size1
    num_cubes_x = int(np.ceil((max_x - min_x) / cube_size))
    num_cubes_y = int(np.ceil((max_y - min_y) / cube_size))
    num_cubes_z = int(np.ceil((max_z - min_z) / cube_size))

    print(f"Total number cubes: {num_cubes_x} (x) * {num_cubes_y} (y) * {num_cubes_z} (z) = {num_cubes_x*num_cubes_y*num_cubes_z}")

    # Iterate over the cubes and save the points in each non-empty cube as a separate .xyz file
    for i in range(num_cubes_x):
        print(f" {i} / {num_cubes_x} (x)")
        for j in range(num_cubes_y):
            for k in range(num_cubes_z):
                # Define the cube boundaries
                cube_min_x = min_x + i * cube_size
                cube_max_x = cube_min_x + cube_size
                cube_min_y = min_y + j * cube_size
                cube_max_y = cube_min_y + cube_size
                cube_min_z = min_z + k * cube_size
                cube_max_z = cube_min_z + cube_size

                # Select the points within the current cube
                mask = (
                    (point_cloud[:, 0] >= cube_min_x) & (point_cloud[:, 0] < cube_max_x) &
                    (point_cloud[:, 1] >= cube_min_y) & (point_cloud[:, 1] < cube_max_y) &
                    (point_cloud[:, 2] >= cube_min_z) & (point_cloud[:, 2] < cube_max_z)
                )
                points_in_cube = point_cloud[mask]

                # Save the points as an .xyz file if the cube contains enough (>500) points
                if 500 < len(points_in_cube) < 3000:
                    #save_as_xyz(points_in_cube, i, j, k)
                    # Generate the file name based on the cube indices
                    #filename = f'/cube_{i}_{j}_{k}.xyz'
                    filename = f'/cube_{i}_{j}_{k}.txt'
                    # Save the points as an .xyz file
                    np.savetxt(outpath+filename, points_in_cube[:,0:3], fmt='%.6f', delimiter=' ')
                if  len(points_in_cube) > 3000:
                    random_indices = np.random.choice(len(points_in_cube), size=2000, replace=False)
                    points_in_cube = points_in_cube[random_indices,:]

    # Find the minimum and maximum coordinates but shift it by 0.5 in every direction
    min_x = np.min(point_cloud[:, 0]) + 0.5
    max_x = np.max(point_cloud[:, 0]) + 0.5
    min_y = np.min(point_cloud[:, 1]) + 0.5
    max_y = np.max(point_cloud[:, 1]) + 0.5
    min_z = np.min(point_cloud[:, 2]) + 0.5
    max_z = np.max(point_cloud[:, 2]) + 0.5

    
    print(f"Cutting with size 2 ({size2})")

    # Calculate the number of cubes in each dimension
    cube_size = size2
    num_cubes_x = int(np.ceil((max_x - min_x) / cube_size))
    num_cubes_y = int(np.ceil((max_y - min_y) / cube_size))
    num_cubes_z = int(np.ceil((max_z - min_z) / cube_size))

    print(f"Total number cubes: {num_cubes_x} (x) * {num_cubes_y} (y) * {num_cubes_z} (z) = {num_cubes_x*num_cubes_y*num_cubes_z}")

    # Iterate over the cubes and save the points in each non-empty cube as a separate .xyz file
    for i in range(num_cubes_x):
        print(f" {i} / {num_cubes_x} (x)")
        for j in range(num_cubes_y):
            for k in range(num_cubes_z):
                # Define the cube boundaries
                cube_min_x = min_x + i * cube_size
                cube_max_x = cube_min_x + cube_size
                cube_min_y = min_y + j * cube_size
                cube_max_y = cube_min_y + cube_size
                cube_min_z = min_z + k * cube_size
                cube_max_z = cube_min_z + cube_size

                # Select the points within the current cube
                mask = (
                    (point_cloud[:, 0] >= cube_min_x) & (point_cloud[:, 0] < cube_max_x) &
                    (point_cloud[:, 1] >= cube_min_y) & (point_cloud[:, 1] < cube_max_y) &
                    (point_cloud[:, 2] >= cube_min_z) & (point_cloud[:, 2] < cube_max_z)
                )
                points_in_cube = point_cloud[mask]

                # Save the points as an .xyz file if the cube contains enough (>500) points
                if 500 < len(points_in_cube) < 4000:
                    #save_as_xyz(points_in_cube, i, j, k)
                    # Generate the file name based on the cube indices
                    #filename = f'cubes/cube_{i}_{j}_{k}.xyz'
                    filename = f'/cube_{i}_{j}_{k}_v2.txt'
                    # Save the points as an .xyz file
                    np.savetxt(outpath+filename, points_in_cube[:,0:3], fmt='%.6f', delimiter=' ')
                if  len(points_in_cube) > 4000:
                    random_indices = np.random.choice(len(points_in_cube), size=2000, replace=False)
                    points_in_cube = points_in_cube[random_indices,:]    
                    filename = f'/cube_{i}_{j}_{k}_v2.txt'
                    # Save the points as an .xyz file
                    np.savetxt(outpath+filename, points_in_cube[:,0:3], fmt='%.6f', delimiter=' ')

    # Find the minimum and maximum coordinates but shift it by 0.5 in every direction
    min_x = np.min(point_cloud[:, 0]) - 0.5
    max_x = np.max(point_cloud[:, 0]) - 0.5
    min_y = np.min(point_cloud[:, 1]) - 0.5
    max_y = np.max(point_cloud[:, 1]) - 0.5
    min_z = np.min(point_cloud[:, 2]) - 0.5
    max_z = np.max(point_cloud[:, 2]) - 0.5

    print(f"Cutting with size 3 ({size3})")

    # Calculate the number of cubes in each dimension
    cube_size = size3
    num_cubes_x = int(np.ceil((max_x - min_x) / cube_size))
    num_cubes_y = int(np.ceil((max_y - min_y) / cube_size))
    num_cubes_z = int(np.ceil((max_z - min_z) / cube_size))

    print(f"Total number cubes: {num_cubes_x} (x) * {num_cubes_y} (y) * {num_cubes_z} (z) = {num_cubes_x*num_cubes_y*num_cubes_z}")

    # Iterate over the cubes and save the points in each non-empty cube as a separate .xyz file
    for i in range(num_cubes_x):
        print(f" {i} / {num_cubes_x} (x)")
        for j in range(num_cubes_y):
            for k in range(num_cubes_z):
                # Define the cube boundaries
                cube_min_x = min_x + i * cube_size
                cube_max_x = cube_min_x + cube_size
                cube_min_y = min_y + j * cube_size
                cube_max_y = cube_min_y + cube_size
                cube_min_z = min_z + k * cube_size
                cube_max_z = cube_min_z + cube_size

                # Select the points within the current cube
                mask = (
                    (point_cloud[:, 0] >= cube_min_x) & (point_cloud[:, 0] < cube_max_x) &
                    (point_cloud[:, 1] >= cube_min_y) & (point_cloud[:, 1] < cube_max_y) &
                    (point_cloud[:, 2] >= cube_min_z) & (point_cloud[:, 2] < cube_max_z)
                )
                points_in_cube = point_cloud[mask]

                # Save the points as an .xyz file if the cube is not empty
                if 1000 < len(points_in_cube) < 4000:
                    #save_as_xyz(points_in_cube, i, j, k)
                    # Generate the file name based on the cube indices
                    #filename = f'cubes/cube_{i}_{j}_{k}.xyz'
                    filename = f'/cube_{i}_{j}_{k}_v3.txt'
                    # Save the points as an .xyz file
                    np.savetxt(outpath+filename, points_in_cube[:,0:3], fmt='%.6f', delimiter=' ')
                if  len(points_in_cube) > 4000:
                    random_indices = np.random.choice(len(points_in_cube), size=2000, replace=False)
                    points_in_cube = points_in_cube[random_indices,:]
                    filename = f'/cube_{i}_{j}_{k}_v3.txt'
                    # Save the points as an .xyz file
                    np.savetxt(outpath+filename, points_in_cube[:,0:3], fmt='%.6f', delimiter=' ')
                    
    # Find the minimum and maximum coordinates but shift it by 0.5 in every direction
    min_x = np.min(point_cloud[:, 0]) 
    max_x = np.max(point_cloud[:, 0])
    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])
    min_z = np.min(point_cloud[:, 2]) - 0.3
    max_z = np.max(point_cloud[:, 2]) - 0.3

    print(f"Cutting with size 4 ({size4})")

    # Calculate the number of cubes in each dimension
    cube_size = size4
    num_cubes_x = int(np.ceil((max_x - min_x) / cube_size))
    num_cubes_y = int(np.ceil((max_y - min_y) / cube_size))
    num_cubes_z = int(np.ceil((max_z - min_z) / cube_size))
    
    print(f"Total number cubes: {num_cubes_x} (x) * {num_cubes_y} (y) * {num_cubes_z} (z) = {num_cubes_x*num_cubes_y*num_cubes_z}")

    # Iterate over the cubes and save the points in each non-empty cube as a separate .xyz file
    for i in range(num_cubes_x):
        print(f" {i} / {num_cubes_x} (x)")
        for j in range(num_cubes_y):
            for k in range(num_cubes_z):
                # Define the cube boundaries
                cube_min_x = min_x + i * cube_size
                cube_max_x = cube_min_x + cube_size
                cube_min_y = min_y + j * cube_size
                cube_max_y = cube_min_y + cube_size
                cube_min_z = min_z + k * cube_size
                cube_max_z = cube_min_z + cube_size

                # Select the points within the current cube
                mask = (
                    (point_cloud[:, 0] >= cube_min_x) & (point_cloud[:, 0] < cube_max_x) &
                    (point_cloud[:, 1] >= cube_min_y) & (point_cloud[:, 1] < cube_max_y) &
                    (point_cloud[:, 2] >= cube_min_z) & (point_cloud[:, 2] < cube_max_z)
                )
                points_in_cube = point_cloud[mask]

                # Save the points as an .xyz file if the cube contains enough (>1000) points
                if 1000 < len(points_in_cube) < 6000:
                    #save_as_xyz(points_in_cube, i, j, k)
                    # Generate the file name based on the cube indices
                    filename = f'/cube_{i}_{j}_{k}_v4.txt'
                    # Save the points as an .xyz file
                    np.savetxt(outpath+filename, points_in_cube[:,0:3], fmt='%.6f', delimiter=' ')                
                if  len(points_in_cube) > 6000:
                    random_indices = np.random.choice(len(points_in_cube), size=2200, replace=False)
                    points_in_cube = points_in_cube[random_indices,:]
                    filename = f'/cube_{i}_{j}_{k}_v4.txt'
                    # Save the points as an .xyz file
                    np.savetxt(outpath+filename, points_in_cube[:,0:3], fmt='%.6f', delimiter=' ')


