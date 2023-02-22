import os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import cv2

from visualizer import Visualizer
from networks import *

import nilearn as nl
import nibabel as nib

#Path to the train dataset.
path_train = "/home/m/Documents/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"

#Path to the validation dataset
path_validation = "/home/m/Documents/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_ValidationData/"

list_dir = os.listdir(path_train)

def getIndex(path_train):
    list_dir = os.listdir(path_train)
    idx = [i.split('_')[-1] for i in list_dir]
    return idx

def loaddata(sample_data):
    sample_img = nib.load(sample_data)
    sample_img = np.asanyarray(sample_img.dataobj)
    sample_img = np.rot90(sample_img)
    #print(sample_img.shape)
    return sample_img


def randomSample(idx):
    global path_train
    list_dir = os.listdir(path_train)

    plot_list = []
    store_path = []
    for i in list_dir:
        data = os.path.join(path_train+i)

        if data.endswith("BraTS20_Training_"+str(idx)):

            for i in sorted(os.listdir(data)):
                if i.endswith("flair.nii"):
                    sample_filename_flair = os.path.join(path_train+ "BraTS20_Training_"+str(idx) + '/' + i)
                    store_path.append(sample_filename_flair)
                    sample_filename_flair = loaddata(sample_filename_flair)
                    plot_list.append(sample_filename_flair)

                if i.endswith("t1.nii"):
                    sample_filename_t1 = os.path.join(path_train+ "BraTS20_Training_"+str(idx) + '/' + i)
                    store_path.append(sample_filename_t1)
                    sample_filename_t1 = loaddata(sample_filename_t1)
                    plot_list.append(sample_filename_t1)

                if i.endswith("t1ce.nii"):
                    sample_filename_t1ce = os.path.join(path_train+ "BraTS20_Training_"+str(idx) + '/' + i)
                    store_path.append(sample_filename_t1ce)
                    sample_filename_t1ce = loaddata(sample_filename_t1ce)
                    plot_list.append(sample_filename_t1ce)

                if i.endswith("t2.nii"):
                    sample_filename_t2 = os.path.join(path_train+ "BraTS20_Training_"+str(idx) + '/' + i)
                    store_path.append(sample_filename_t2)
                    sample_filename_t2 = loaddata(sample_filename_t2)
                    plot_list.append(sample_filename_t2)

                if i.endswith("seg.nii"):
                    sample_filename_mask = os.path.join(path_train+ "BraTS20_Training_"+str(idx) + '/' + i)
                    store_path.append(sample_filename_mask)
                    sample_filename_mask = loaddata(sample_filename_mask)
                    plot_list.append(sample_filename_mask)

    return plot_list, store_path

# plot_list consists of list of image types.
# plot_list[0] - flair
# plot_list[1] - seg
# plot_list[2] - t1
# plot_list[3] - t1ce
# plot_list[4] - t2

plot_list, store_path = randomSample('150')

# instantiation
#viz = Visualizer(plot_list[2], plot_list, store_path[0])
#viz.three_d_array_to_gif(store_path[0])



