# Research-on-Deep-Denoising

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Pytorch Version
#### Dependencies

This tutorial depends on the following libraries:

* Pytorch
* Numpy

#### Training

The model is trained for >200 epochs.

#### Testing



##### Results

Use the trained model to do segmentation on test images, the result is statisfactory.

![demo/Test0_data.png](img/0test.png)

![img/Test0_label.png](img/0label.png)

### Numpy Version
#### Dependencies
* Numpy
* skimage
* h5py

#### Coverting pretrained model into hf files
python pth2h5.py

### Testing
python ./Numpy_V2/Numpy_test.py

#### Results
