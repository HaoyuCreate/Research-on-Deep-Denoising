# Research-on-Deep-Denoising

Neural networks have achieved superior performance in various important tasks. However, lack of the interpretability makes the neural networks still a black box for most of the users. The black-box nature becomes the Achille's heel of neural network for full-exploitation. Triggered by these challenges, I focus on signal denoising tasks and propose an end-to-end-trained network guided by task-based knowledge that transforms input data into an integrated and ordered representative in high-dimensional feature space.

Moreover, to understand how a deep neural network (DNN) performs like a filter in the signal denoising task, I re-implemented a forward branch of the testing process of the forementioned network by Numpy only. While implementing the network, I also obverse some interesting facts of the DNN which triggers me to have some new thoughts (i.e. batch normalization layer with adjustable sparse regularization) on improving DNN. In the future study, I will try to conclude a general form of unsupervised deep neural networks for denoising tasks, which are more close to blind denoising approaches in practice. Besides, many advanced adaptive filtering algorithms with sparse regularization will be tested to take the place of the traditional batch normalization layers and further improve the capability of deep neural networks.

---

## Overview

### 1. Pytorch Version
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
<img src="demo/Test0_Weiner_pred.png" alt="Wiener pred" width="502"/>
<img src="demo/Test0_Torch_pred_Unfixed_batch.png" alt="Unfixed pred" width="502"/>
<img src="demo/Test0_Numpy_L1_pred.png" alt="Li-norm pred" width="502"/>
<img src="demo/Test0_Torch_pred_Fixed_batch.png" alt="Unfixed pred" width="502"/>



### 2. Numpy Version
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
<img src="demo/Test1_Weiner_pred.png" alt="Wiener pred" width="502"/>
<img src="demo/Test1_Numpy_pred_Unfixed_batch.png" alt="Unfixed pred" width="502"/>
<img src="demo/Test1_Numpy_L1_pred.png" alt="Li-norm pred" width="502"/>
<img src="demo/Test1_Numpy_pred_Fixed_batch.png" alt="Unfixed pred" width="502"/>

