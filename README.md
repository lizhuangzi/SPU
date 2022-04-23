# Semantic Point Cloud Upsampling (SPU)
Code of Semantic Point Cloud Upsampling (SPU) published on IEEE Transactions on Multimedia.

## Abstract
Downsampled sparse point clouds are beneficial for data transmission and storage, but they are detrimental for semantic tasks due to information loss. In this paper, we examine an upsampling methodology that significantly reconstructs sparse clouds' semantic representations. Specifically, we propose a novel semantic point cloud upsampling (SPU) framework for sparse point cloud classification. An SPU consists of two networks, i.e. an upsampling network and a classification network. They are skillfully unified to intensify semantic representations acting on the upsampling process. In the upsampling network, we first propose a novel graph aggregation convolution to construct hierarchical relations on sparse point clouds. To enhance stability and diversity during point upsampling, we then combine point shuffling and pre-interpolation technologies to build an enhanced upsampling module.
Furthermore, we adopt the semantic prior information provided by a sparse point cloud to enhance its upsampling quality. 
The prior information is applied to an attention mechanism that can highlight key positions of the point cloud.
We investigate different loss functions and conduct experiments on classical deep point networks, which effectively demonstrate the promising performance of our framework.


![image](https://github.com/lizhuangzi/SPU/blob/main/Fig/1.png)

![image](https://github.com/lizhuangzi/SPU/blob/main/Fig/2.png)

## Requirements (Our setting)
Python 3.6

PyTorch 1.2.0

Torchvision 0.4.0

KNN_CUDA 0.2.0


## Pretrained Model
You can get the trained upsampling model in the dir of "./savedModel".

The pre-trained classification networks are at "./PretrainModel"


## Usage

Using "cd ./train" and then Using "python trainSPUDGCNN8x.py" or "python trainSPUPointNet8x.py" for training.

The trainSPUDGCNN8x.py is based on the classification network of DGCNN and the trainSPUPointNet8x.py is based the classification network of PointNet.
