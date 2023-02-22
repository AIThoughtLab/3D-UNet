
# This is a data generator class

import numpy as np
import os
import nibabel as nib
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from keras.optimizers import Adam
from scipy.ndimage import gaussian_filter, laplace
import SimpleITK as sitk
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=1, dim=(144, 224, 224), n_channels=4, n_classes=4, shuffle=True,
                 augment=True):
        self.data_dir = data_dir
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.files = os.listdir(data_dir)
        self.on_epoch_end()
        self.image_datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True)
        # self.is_training = is_training

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        files = [self.files[i] for i in indexes]
        return self.__data_generation(files)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def crop(self, mri_data, data_mri=True):

        # Define the cropping parameters
        start = (6, 8, 8)
        end = (-5, -8, -8)

        # Crop the MRI data using the defined parameters
        data_cropped = mri_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        # return data_cropped
        if data_mri:
            min_ = np.min(data_cropped)
            data_cropped = (data_cropped - min_) / (np.max(data_cropped) - min_)
            data_cropped = np.round(data_cropped, 3)
            # print(data_cropped.shape)

            data_cropped = gaussian_filter(data_cropped, sigma=(1, 1, 1))  # Apply Gaussian smoothing to reduce noise
            data_cropped = data_cropped - 0.3 * laplace(data_cropped)  # Apply the Laplacian filter to sharpen the image
            # print("unique image values: ", np.unique(data_cropped))

            return data_cropped

        else:
            return data_cropped

    def __data_generation(self, files):

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)  # input
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=np.float32)  # output/labels

        for i, file in enumerate(files):

            if not file.endswith(".csv"):

                directory = os.path.join(self.data_dir, file)
                flair = nib.load(os.path.join(directory, file + "_flair.nii")).get_fdata()
                flair = np.moveaxis(flair, [2, 0, 1], [0, 1, 2])
                flair = self.crop(flair, data_mri=True)
                # print("flair", flair.shape)

                t1 = nib.load(os.path.join(directory, file + "_t1.nii")).get_fdata()
                t1 = np.moveaxis(t1, [2, 0, 1], [0, 1, 2])
                t1 = self.crop(t1, data_mri=True)

                t1ce = nib.load(os.path.join(directory, file + "_t1ce.nii")).get_fdata()
                t1ce = np.moveaxis(t1ce, [2, 0, 1], [0, 1, 2])
                t1ce = self.crop(t1ce, data_mri=True)

                t2 = nib.load(os.path.join(directory, file + "_t2.nii")).get_fdata()
                t2 = np.moveaxis(t2, [2, 0, 1], [0, 1, 2])
                t2 = self.crop(t2, data_mri=True)

                # load the mask
                mask = nib.load(os.path.join(directory, file + "_seg.nii")).get_fdata()
                mask = np.moveaxis(mask, [2, 0, 1], [0, 1, 2])

                mask[mask == 4] = 3
                # print('unique mask before croping: ', np.unique(mask))

                mask = self.crop(mask, data_mri=False)
                # print('unique mask after croping: ', np.unique(mask))

                mask = mask.astype('int')  # convert mask to integer data type
                # print("mask shape: ", mask.shape)

                num_classes = 4
                mask = np.expand_dims(mask, axis=-1)
                masks = tf.keras.utils.to_categorical(mask, num_classes)

                # create stack for images and one-hot for masks..
                images = np.stack([flair, t1, t1ce, t2], axis=-1)

                if self.augment:
                    # apply random rotation and flip
                    seed = np.random.randint(1, 100)
                    datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True)

                    for k in range(len(images)):
                        images[k] = datagen.random_transform(images[k], seed=seed)
                        masks[k] = datagen.random_transform(masks[k], seed=seed)
                        masks = masks.astype(int)  # convert to int type so we will have 0 and 1

                    X[i,] = images
                    y[i,] = masks

                else:
                    X[i,] = images
                    y[i,] = masks

        return X, y