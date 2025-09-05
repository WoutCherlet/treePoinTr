import numpy as np

file = "/Stor1/wout/OcclusionPaper/CLS_experiment/all_cubes/COL/blocks/block_000_018_022.npz"

arr = np.load(file)

if file[-4:] == ".npz":
    for array in arr:
        print(array)
        print(arr[array].shape)
else:
    print(arr.shape)
    print(arr[:10])