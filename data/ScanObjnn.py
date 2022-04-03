#!/usr/bin/env python
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import utils.data_util as utils




def load_data(filename, partition):
    f = h5py.File(filename)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    all_data = np.array(data)
    label = np.array(label)

    return all_data,label



def translate_pointcloud(pointcloud):
    """
    for scaling and shifting the point cloud
    :param pointcloud:
    :return:
    """
    scale = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, scale), shift).astype('float32')
    return translated_pointcloud


class ScanObjNN(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    def __init__(self, num_points=1024, data_dir="", partition='train',use_norm=True):
        if partition == 'train':
            data_dir = '/mnt/cloud_disk/LZZData/h5_files/h5_files/main_split/training_objectdataset.h5'
        else:
            data_dir = '/mnt/cloud_disk/LZZData/h5_files/h5_files/main_split/test_objectdataset.h5'

        self.data, self.label = load_data(data_dir, partition)
        self.num_points = num_points
        self.partition = partition

        centroid = np.mean(self.data[..., :3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((self.data[..., :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        self.radius = furthest_distance[:, 0]  # not very sure?

        #norm
        self.radius = np.ones(shape=(len(self.data)))
        self.data[..., :3] -= centroid
        self.data[..., :3] /= np.expand_dims(furthest_distance, axis=-1)
        self.nrepeat = 3
        print('aa')


    def __getitem__(self, item):

        if self.partition == 'train':
            totalnumber = self.data.shape[0]
            item = item % totalnumber

        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud,item, label

    def __len__(self):
        if self.partition == 'train':
            return self.data.shape[0]*self.nrepeat
        else:
            return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1


    # def __getitem__(self, item):
    #     pointcloud = self.data[item][:self.num_points]
    #     if self.upscalefactor>1:
    #         sample_idx = utils.nonuniform_sampling(self.num_points,sample_num=self.num_points//self.upscalefactor)
    #         lrpoint = pointcloud[sample_idx, :]
    #     else:
    #         lrpoint = pointcloud
    #     label = self.label[item]
    #
    #     if self.partition == 'test':
    #         return lrpoint,pointcloud,label

class ScanObjNNSPU(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    def __init__(self, num_points=1024, data_dir="", partition='train',use_norm=True,upscalefactor=4):
        if partition == 'train':
            data_dir = '/mnt/cloud_disk/LZZData/h5_files/h5_files/main_split/training_objectdataset.h5'
        else:
            data_dir = '/mnt/cloud_disk/LZZData/h5_files/h5_files/main_split/test_objectdataset.h5'

        self.data, self.label = load_data(data_dir, partition)
        self.num_points = num_points
        self.partition = partition
        self.upscalefactor = upscalefactor

        centroid = np.mean(self.data[..., :3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((self.data[..., :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        self.radius = furthest_distance[:, 0]  # not very sure?

        #norm
        self.radius = np.ones(shape=(len(self.data)))
        self.data[..., :3] -= centroid
        self.data[..., :3] /= np.expand_dims(furthest_distance, axis=-1)
        self.nrepeat = 1
        print('aa')


    def __getitem__(self, item):

        if self.partition == 'train':
            totalnumber = self.data.shape[0]
            item = item % totalnumber


        pointcloud = self.data[item][:self.num_points]
        if self.upscalefactor != 1:
            sample_idx = utils.nonuniform_sampling(self.num_points,sample_num=self.num_points//self.upscalefactor)
            lrpoint = pointcloud[sample_idx, :]
        else:
            lrpoint = pointcloud
        label = self.label[item]

        if self.partition == 'test':
            return lrpoint,pointcloud,label

        input_data, gt_data = utils.rotate_point_cloud_and_gt(lrpoint, pointcloud)
        input_data, gt_data, scale = utils.random_scale_point_cloud_and_gt(input_data, gt_data,
                                                                           scale_low=0.9, scale_high=1.1)
        input_data, gt_data = utils.shift_point_cloud_and_gt(input_data, gt_data, shift_range=0.1)

        return input_data, gt_data,label

    def __len__(self):
        if self.partition == 'train':
            return self.data.shape[0]
        else:
            return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1


    # def __getitem__(self, item):
    #     pointcloud = self.data[item][:self.num_points]
    #     if self.upscalefactor>1:
    #         sample_idx = utils.nonuniform_sampling(self.num_points,sample_num=self.num_points//self.upscalefactor)
    #         lrpoint = pointcloud[sample_idx, :]
    #     else:
    #         lrpoint = pointcloud
    #     label = self.label[item]
    #
    #     if self.partition == 'test':
    #         return lrpoint,pointcloud,label

if __name__ == '__main__':
    train = ScanObjNNSPU(num_points=1024,partition='train')
    print(train[0])


