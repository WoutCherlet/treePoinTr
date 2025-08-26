# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:34:23 2023

@author: Bornand
"""

import numpy as np
#import open3d as o3d
import os
import glob
#import laspy
import random
#import re
import pandas as pd
import sys
#from sklearn.model_selection import train_test_split

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


def check_string_contains(species_list, target_string):
    for species in species_list:
        if species in target_string:
            return species
        else:
            return "undefined"





def mksamples(inpath, outpath, start_count=0, stop_count=100, boxsize=0.5, method = 'grove'):
    """ 
    start_count, stop_count: sample number
    boxsize: edge length of 3D sample box
    method: description of data source, e.g. created using "the Grove"
    """
    file = os.listdir(inpath)
    
    for item in file:

        #treenumber = re.findall(r'\d+', item)

        treename = os.path.splitext(os.path.basename(item))[0]
        
        # get species if available
        species_list = ["ash", "beech", "elm", "linden", "maple", "oak", "walnut"]
        species = check_string_contains(species_list, treename)
        
        # load point cloud and store as numpy array
        if item.endswith('.npy'):
            pc = np.load(inpath+item)
            
        if item.endswith('.xyz'):
            pc = np.loadtxt(inpath+item, skiprows=1, delimiter=",")
            
        if item.endswith('.txt'):
           pc = np.loadtxt(inpath+item, skiprows=1, delimiter=",") 
            
        if item.endswith('.ply'):
            ply_cloud = o3d.io.read_point_cloud(inpath+item) 
            pc = np.asarray(ply_cloud.points)
            
        # if item.endswith('.las'):
        #    las_cloud = laspy.read(inpath+item)
        #    pc = las_cloud.points

        
        # # for data from tree generator: convert from mm to m
        # pc[:, 0] = pc[:, 0]/1000
        # pc[:, 1] = pc[:, 1]/1000
        # pc[:, 2] = pc[:, 2]/1000
        
        # for real data (from SwissBiomass): translate point cloud into fully positive values 
        # pc[:, 0] = pc[:, 0]+100
        # pc[:, 1] = pc[:, 1]+100
        
        
        
        
        # initialization
        count = start_count
        # number of samples:
        n = stop_count
        
        while count in range(n):
            # select a random point (row)
            number_of_rows = pc.shape[0]
            random_indices = np.random.choice(number_of_rows, size=1, replace=False)
            # display random rows
            #print("\nRandom row:")
            row = pc[random_indices, :]
            #print(row)
            
            
            # bounding box selection around point
            s = boxsize

            min_x = row[0,0]-s
            max_x = row[0,0]+s
            min_y = row[0,1]-s
            max_y = row[0,1]+s
            min_z = row[0,2]-s
            max_z = row[0,2]+s
            
            bb_filter = bounding_box(pc, min_x, max_x, min_y, max_y, min_z, max_z)
            
            sample = pc[bb_filter,:]
            
            if abs(min(sample[:, 0])) > 100:
               sample[:, 0] = sample[:, 0] + (0-min(sample[:, 0]))
               sample[:, 1] = sample[:, 1] + (0-min(sample[:, 1]))
               sample[:, 2] = sample[:, 2] + (0-min(sample[:, 2]))

            
            if len(sample)>8191:
            
                # for real data: translate sample point cloud back into original
                # sample[:, 0] = sample[:, 0]-100
                # sample[:, 1] = sample[:, 1]-100
            
                # downsample to 8192 points
                number_of_rows = sample.shape[0]
                random_indices = np.random.choice(number_of_rows, size=8192, replace=False)
                sample = sample[random_indices,:]
                
                # save sample as .npy
                outname = method+'-'+species+'-'+treename+'_'+str(count)+'_size'+str(s)+'.npy' 
                np.save(outpath+outname, sample[:,0:3])

                count += 1


def mksamples_array(inpath, outpath, start_count=0, stop_count=100, boxsize=0.5, method = 'grove'):
    """ 
    start_count, stop_count: sample number
    boxsize: edge length of 3D sample box
    method: description of data source, e.g. created using "the Grove"
    """
    items = os.listdir(inpath)
    print(int(sys.argv[1]))
    item = items[int(sys.argv[1])]


    #treenumber = re.findall(r'\d+', item)

    treename = os.path.splitext(os.path.basename(item))[0]
    print(treename)
    # get species if available
    species = "na"
    species_list = ["ash", "beech", "elm", "linden", "maple", "oak", "walnut"]
    #species = check_string_contains(species_list, treename)
    for name in species_list:
        if name in treename:
            species=name
        #else:
            #species = "undefined"
        
            
	
    #print(species)
    #species = "na"

    
    # load point cloud and store as numpy array
    if item.endswith('.npy'):
        pc = np.load(inpath+item)
        
    if item.endswith('.xyz'):
        pc = np.loadtxt(inpath+item, skiprows=1, delimiter=",")
        
    if item.endswith('.txt'):
       pc = np.loadtxt(inpath+item, skiprows=1, delimiter=",") 
        
    if item.endswith('.ply'):
        ply_cloud = o3d.io.read_point_cloud(inpath+item) 
        pc = np.asarray(ply_cloud.points)
        
    # if item.endswith('.las'):
    #    las_cloud = laspy.read(inpath+item)
    #    pc = las_cloud.points

    
    # # for data from tree generator: convert from mm to m
    # pc[:, 0] = pc[:, 0]/1000
    # pc[:, 1] = pc[:, 1]/1000
    # pc[:, 2] = pc[:, 2]/1000
    
    # for real data (from SwissBiomass): translate point cloud into fully positive values 
    # pc[:, 0] = pc[:, 0]+100
    # pc[:, 1] = pc[:, 1]+100
    
    number_of_rows = pc.shape[0]
    
    # initialization
    count = start_count
    # number of samples:
    n = stop_count
    
    while count in range(n):
        # select a random point (row)
        random_indices = np.random.choice(number_of_rows, size=1, replace=False)
        # display random rows
        #print("\nRandom row:")
        row = pc[random_indices, :]
        #print(row)
        
        
        # bounding box selection around point
        s = boxsize

        min_x = row[0,0]-s
        max_x = row[0,0]+s
        min_y = row[0,1]-s
        max_y = row[0,1]+s
        min_z = row[0,2]-s
        max_z = row[0,2]+s
        
        bb_filter = bounding_box(pc, min_x, max_x, min_y, max_y, min_z, max_z)
        
        sample = pc[bb_filter,:]
        
        if abs(min(sample[:, 0])) > 100:
           sample[:, 0] = sample[:, 0] + (0-min(sample[:, 0]))
           sample[:, 1] = sample[:, 1] + (0-min(sample[:, 1]))
           sample[:, 2] = sample[:, 2] + (0-min(sample[:, 2])) 
        
        if len(sample)>8191:
        
            # for real data: translate sample point cloud back into original
            # sample[:, 0] = sample[:, 0]-100
            # sample[:, 1] = sample[:, 1]-100
        
            # downsample to 8192 points
            number_of_rows = sample.shape[0]
            random_indices = np.random.choice(number_of_rows, size=8192, replace=False)
            sample = sample[random_indices,:]
            
            # save sample as .npy
            outname = method+'-'+species+'-'+treename+'_'+str(count)+'_size'+str(s)+'.npy' 
            np.save(outpath+outname, sample[:,0:3])

            count += 1



def train_test_txt(path_samples, outpath, test_trees = True):
    npyfiles = glob.glob(path_samples +"*.npy")
    object_ids =[]
    for file in npyfiles:
        object_id = file.split('-')[2].split('_')[0]
        object_ids.append(object_id)
    df = pd.DataFrame({'object_id': object_ids, 'filename': npyfiles}) #, columns=['object_id', 'filename']
    
    if test_trees:
        test_ids = ["ash3", "beech4", "elm3", "linden4", "maple5", "oak4", "43059", "431362", "8407"]
        
        test = df[df['object_id'].isin(test_ids)] 
        train = df[~df['object_id'].isin(test_ids)]

        file = open(outpath+'test.txt','w')
        for item in test.filename:
            file.write(os.path.basename(item)+"\n")
        file.close()
    
        file = open(outpath+'train.txt','w')
        for item in train.filename:
            file.write(os.path.basename(item)+"\n")
        file.close()
    
    if not test_trees:
        test = random.sample(npyfiles, int(len(npyfiles) * 0.2))
        train = np.setdiff1d(npyfiles,test)
        
        file = open(outpath+'test.txt','w')
        for item in test:
            file.write(os.path.basename(item)+"\n")
        file.close()
    
        file = open(outpath+'train.txt','w')
        for item in train:
            file.write(os.path.basename(item)+"\n")
        file.close()
        
        # split the list of object ids
    # ids = list(set(object_ids))
    # test_ids = random.sample(ids, int(len(ids) * 0.2))
  
    # Splitting the data into train and test sets with stratification based on object_id
    #train, test = train_test_split(df, test_size=0.2, stratify=df['object_id'], random_state=42)
  

    
    
