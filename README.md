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

python ./Pytorch_V1/train.py

#### Testing
python ./Pytorch_V1/test.py


##### Results

Use the trained model to denoise self-synthesied noisy 1D signals. Network is programed by Pytorch.
<img src="demo/Test0_data.png" alt="Test0 data" width="500"/>
<img src="demo/Test0_label.png" alt="Test0 label" width="500"/>
<img src="demo/Test0_Torch_pred_Unfixed_batch.png" alt="Unfixed pred" width="505"/>
<img src="demo/Test0_Weiner_pred.png" alt="Wiener pred" width="505"/>
<img src="demo/Test0_Torch_pred_Fixed_batch.png" alt="Unfixed pred" width="505"/>
<img src="demo/Test0_Weiner_pred.png" alt="Fixed pred" width="505"/>
<img src="demo/Test0_Numpy_L1_pred.png" alt="Li-norm pred" width="505"/>


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
Use the trained model to denoise self-synthesied noisy 1D signals. Network is programed by Numpy.
<img src="demo/Test1_data.png" alt="Test0 data" width="500"/>
<img src="demo/Test1_label.png" alt="Test0 label" width="500"/>
<img src="demo/Test1_Numpy_pred_Unfixed_batch.png" alt="Unfixed pred" width="520"/>
<img src="demo/Test1_Weiner_pred.png" alt="Wiener pred" width="520"/>
<img src="demo/Test1_Numpy_pred_Fixed_batch.png" alt="Unfixed pred" width="520"/>
<img src="demo/Test1_Weiner_pred.png" alt="Fixed pred" width="520"/>
<img src="demo/Test1_Numpy_L1_pred.png" alt="Li-norm pred" width="520"/>
