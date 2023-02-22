
**3D UNet for Brain Tumor Segmentation**

This repository was created to showcase the application of 3D UNet to the brats2020 dataset. Instead of using traditional 2D slices, we fed the image as a 3D dimensional volume. To make the process easier, we divided the training and validation data into separate folders.

To use this repository, 
* run "pip install -r requirements.txt" in your shell and set the path according to your data location. 
* Please note that this repository has been tested in Google Colab Pro.

If you want to test, start at **brats2020.ipynb** and execute each cell.

A mean-iou of 91.94 % obtained in the validation dataset. 
For more information visit [here](https://aithoughtlab.github.io/ThinkingAI/projects/3d-unet/)
