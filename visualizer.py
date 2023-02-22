
# This class is for visualization purposes along the x, y and z axis.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import data
from skimage.util import montage
import skimage.transform as skTrans
from skimage.transform import rotate
import nifti2gif.core as nifti2gif
from skimage.transform import resize
import copy
import shutil
from PIL import Image
from IPython.display import Image

class Visualizer:
    def __init__(self, arr, arrList, gif_path:str):
        # pass a 3D numpy array
        self.arr = arr
        self.arrList = arrList # list of arrays
        self.gif_path = gif_path # string type


    # define a method to plot a slice of the numpy array
    def slice(self, ax, slider_val, axis):
        if axis == 0:
            slice_ = self.arr[int(slider_val), :, :]
        elif axis == 1:
            slice_ = self.arr[:, int(slider_val), :]
        elif axis == 2:
            slice_ = self.arr[:, :, int(slider_val)]
        # plot the slice
        ax.imshow(slice_, cmap='bone')

    def show_slices(self):
        # create the figure and axis objects
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # add the x-axis slider to the figure
        slider_ax1 = plt.axes([0.1, 0.05, 0.2, 0.02])
        slider1 = Slider(slider_ax1, 'X Slice', 0, 239, valinit=0, valstep=1)
        slider1.on_changed(lambda val: self.slice(ax1, val, 0))

        # add the y-axis slider to the figure
        slider_ax2 = plt.axes([0.4, 0.05, 0.2, 0.02])
        slider2 = Slider(slider_ax2, 'Y Slice', 0, 239, valinit=0, valstep=1)
        slider2.on_changed(lambda val: self.slice(ax2, val, 1))

        # add the z-axis slider to the figure
        slider_ax3 = plt.axes([0.7, 0.05, 0.2, 0.02])
        slider3 = Slider(slider_ax3, 'Z Slice', 0, 154, valinit=0, valstep=1)
        slider3.on_changed(lambda val: self.slice(ax3, val, 2))

        # plot the first slices
        self.slice(ax1, 0, 0)
        self.slice(ax2, 0, 1)
        self.slice(ax3, 0, 2)

        # display the figure
        plt.show()

    def simpleplot(self, id):
        # pass which slice index to be displayed.
        # arrList is a list of 3D numpy array
        # this will display t1, t1ce, t2, flair and seg
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))

        ax1.imshow(self.arrList[0][:, :, id], cmap='bone')
        ax1.set_title('Image flair')

        ax2.imshow(self.arrList[1][:, :, id], cmap='bone')
        ax2.set_title('Mask')

        ax3.imshow(self.arrList[2][:, :, id], cmap='bone')
        ax3.set_title('Image t1')

        ax4.imshow(self.arrList[2][:, :, id], cmap='bone')
        ax4.set_title('Image t1ce')

        ax5.imshow(self.arrList[2][:, :, id], cmap='bone')
        ax5.set_title('Image t2')
        # You can add plots to the remaining subplots as needed
        plt.show()


    def montage_(self, arr):
        # Visualize between 60 to -60.. meaning we skip the rest.
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
        ax1.imshow(rotate(montage(arr[50:-50, :, :]), 90, resize=True), cmap='bone')
        plt.show()

    def three_d_array_to_gif(self, gif_path):
        # This function helps to create a gif.
        # Pass a path as a string format.
        # create a copy of the gif path
        shutil.copy2(gif_path, './copy.nii')

        # Write the image as a GIF with normal intensity scaling
        nifti2gif.write_gif_normal('./copy.nii')




