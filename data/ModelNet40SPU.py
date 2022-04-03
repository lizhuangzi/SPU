#!/usr/bin/env python
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import utils.data_util as utils

def download(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], data_dir))
        os.system('rm %s' % (zipfile))


def load_data(data_dir, partition):
    download(data_dir)
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


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


class ModelNet40spu(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    def __init__(self, num_points=1024, data_dir="./", partition='train',use_norm=True,upscalefactor=4):
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
        print('aa')


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        sample_idx = utils.nonuniform_sampling(self.num_points,sample_num=self.num_points//self.upscalefactor)
        lrpoint = pointcloud[sample_idx, :]
        label = self.label[item]

        if self.partition == 'test':
            return lrpoint,pointcloud,label


        #pointcloud = translate_pointcloud(pointcloud)
        #np.random.shuffle(pointcloud)

        input_data, gt_data = utils.rotate_point_cloud_and_gt(lrpoint, pointcloud)
        input_data, gt_data, scale = utils.random_scale_point_cloud_and_gt(input_data, gt_data,
                                                                           scale_low=0.9, scale_high=1.1)
        input_data, gt_data = utils.shift_point_cloud_and_gt(input_data, gt_data, shift_range=0.1)

        return input_data, gt_data,label

    def __len__(self):
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
class ModelNet40spunoAgu(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    def __init__(self, num_points=1024, data_dir="./", partition='train',use_norm=True,upscalefactor=4):
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
        print('aa')


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        sample_idx = utils.nonuniform_sampling(self.num_points,sample_num=self.num_points//self.upscalefactor)
        lrpoint = pointcloud[sample_idx, :]
        label = self.label[item]

        if self.partition == 'test':
            return lrpoint,pointcloud,label


        #pointcloud = translate_pointcloud(pointcloud)
        #np.random.shuffle(pointcloud)

        return lrpoint, pointcloud,label

    def __len__(self):
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
class ModelNet40_DT(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    def __init__(self, num_points=1024, data_dir="./", partition='train',use_norm=True,upscalefactor=4):
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


    def __len__(self):
        return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.upscalefactor>1:
            sample_idx = utils.nonuniform_sampling(self.num_points,sample_num=self.num_points//self.upscalefactor)
            lrpoint = pointcloud[sample_idx, :]
        else:
            lrpoint = pointcloud
        label = self.label[item]

        if self.partition == 'test':
            return lrpoint,pointcloud,label


class ModelNet40spuNoise(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    def __init__(self, num_points=1024, data_dir="./", partition='train',use_norm=True,upscalefactor=4):
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
        print('aa')


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        sample_idx = utils.nonuniform_sampling(self.num_points,sample_num=self.num_points//self.upscalefactor)
        lrpoint = pointcloud[sample_idx, :]
        label = self.label[item]

        #pointcloud = translate_pointcloud(pointcloud)
        #np.random.shuffle(pointcloud)
        #input_data = utils.rotate_perturbation_point_cloud(lrpoint,angle_sigma=0.1, angle_clip=0.09)
        #input_data, gt_data = utils.rotate_point_cloud_and_gt(lrpoint, pointcloud)
        # input_data, gt_data, scale = utils.random_scale_point_cloud_and_gt(input_data, gt_data,
        #                                                                    scale_low=0.9, scale_high=1.1)
        input_data, _ = utils.shift_point_cloud_and_gt(lrpoint, shift_range=0.7)

        return input_data, pointcloud,label

    def __len__(self):
        return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1

if __name__ == '__main__':
    train = ModelNet40spu(num_points=1024,partition='train')
    test = ModelNet40spu(num_points=1024, partition='test')
    for data, label,label2 in test:
        print(data)

