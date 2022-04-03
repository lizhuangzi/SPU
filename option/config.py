import os
import sys
import argparse
import shutil
import random
import numpy as np
import torch
import logging
import time
import uuid
import pathlib
import glob


class OptInit:
    def __init__(self):
        # ===> argparse
        parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
        # ----------------- Log related
        parser.add_argument('--exp_name', type=str, default='DeepGCN', metavar='N',
                            help='Name of the experiment')
        parser.add_argument('--root_dir', type=str, default='log', help='the dir of experiment results, ckpt and logs')

        # ----------------- Dataset related
        parser.add_argument('--data_dir', type=str, default='../data')
        parser.add_argument('--num_points', type=int, default=1024,
                            help='num of points to use')
        parser.add_argument('--augment', action='store_true', default=True, help='Data Augmentation')

        # ----------------- Training related
        parser.add_argument('--phase', type=str, default='train', metavar='N',
                            choices=['train', 'test'])
        parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu?')
        parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                            help='Size of batch)')
        parser.add_argument('--epochs', type=int, default=200, metavar='N',
                            help='number of episode to train ')
        parser.add_argument('--optimizer', type=bool, default='Adam')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.001)')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--multi_gpus', action='store_true', help='use multi-gpus')

        # ----------------- Testing related
        parser.add_argument('--test_batch_size', type=int, default=50, metavar='batch_size',
                            help='Size of batch)')
        parser.add_argument('--pretrained_model', type=str, default='', metavar='N',
                            help='Pretrained model path')

        # ----------------- Model related
        parser.add_argument('--k', default=9, type=int, help='neighbor num (default:9)')
        parser.add_argument('--block', default='res', type=str, help='graph backbone block type {res, plain, dense}')
        parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
        parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--norm', default='batch', type=str,
                            help='batch or instance normalization {batch, instance}')
        parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
        parser.add_argument('--n_blocks', type=int, default=14, help='number of basic blocks in the backbone')
        parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
        parser.add_argument('--in_channels', type=int, default=3, help='Dimension of input ')
        parser.add_argument('--out_channels', type=int, default=40, help='Dimension of out_channels ')
        parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
        parser.add_argument('--pretrainModel',default='PointNetModelNet40_fitLR.parm')

        # dilated knn
        parser.add_argument('--use_dilation', default=True, type=bool, help='use dilated knn or not')
        parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
        parser.add_argument('--stochastic', default=True, type=bool, help='stochastic for gcn, True or False')

        args = parser.parse_args()
        args.device = torch.device('cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu')
        self.args = args

    def _get_args(self):
        return self.args
