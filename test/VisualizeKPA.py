import numpy as np
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

import sklearn.metrics as metrics
from data import ModelNet40
from tqdm import tqdm
from Clsutils.eval import accuracy
from Clsutils.misc import AverageMeter
from ClsNetwork.PointNetcls import PoiintNet
from ClsNetwork.DGCNNcls import DGCNN_cls
from ArchiAblation.SPUfinal8x import Generator
from option.config import OptInit
from data.ModelNet40SPU import ModelNet40spu
from data.ModelNet40 import ModelNet40_DA
import os
import random
import pandas as pd
from option.train_option import get_train_options
from utils.xyz_util import save_xyz_file
from utils.ply_utils import save_ply
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.pc_util import draw_colorpoint_cloud

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def infer(model,G_model, test_loader, criterion, opt):
    model.eval()
    G_model.eval()
    save_dir = '../vis_result2/KPAout/'


    for i, (data, gt, label) in enumerate(tqdm(test_loader)):
        data, label = data.to('cuda:2'), label.to('cuda:2').squeeze()
        data = data.permute(0, 2, 1)
        #gt = gt.cuda().permute(0, 2, 1)
        #Nearestup = F.upsample_nearest(data, scale_factor=4)

        data.requires_grad = True
        #print(data.shape)
        lr_ClsResult, ff, _ = model(data)
        weight2 = torch.ones(lr_ClsResult.size()).to('cuda:2')
        # weight2 = weight2 / 40
        lrdata_jocabian, = torch.autograd.grad(lr_ClsResult, data, weight2, retain_graph=True)
        lr_jacobian = F.softmin(lrdata_jocabian.mean(dim=1, keepdim=True), dim=2)

        savedpath = save_dir+'%d.jpg'%i
        draw_colorpoint_cloud(data[3].detach().cpu().numpy().transpose(), zrot=90 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,
                                    diameter=10, normalize=True, canvasSize=1000, space=480,weights = lr_jacobian[0],path=savedpath,ifmaskother=False)





if __name__ == '__main__':

    seed = 1234
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

    opt = OptInit()._get_args()
    params = get_train_options()


    #train_loader = DataLoader(ModelNet40(data_dir=opt.data_dir, partition='train', num_points=opt.num_points),
    #                          num_workers=8, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    #test_loader = DataLoader(ModelNet40(data_dir=opt.data_dir, partition='test', num_points=opt.num_points),
    #                         num_workers=4, batch_size=opt.test_batch_size, shuffle=True, drop_last=False)

    eval_dst = ModelNet40spu(data_dir=params["dataset_dir"], partition='test', num_points=1024,upscalefactor=2)
    #eval_dst = ModelNet40_DA(data_dir=params["dataset_dir"], partition='test', num_points=1024,upscalefactor=4)
    test_loader = DataLoader(eval_dst, batch_size=50,
                                 shuffle=False, pin_memory=True, num_workers=4)

    opt.n_classes = 40

    model = PoiintNet(k=opt.n_classes,normal_channel=False)
   # model = DGCNN_cls(opt.n_classes)
    dic = torch.load('../PretrainModel/' + 'PointNetModelNet40.parm', map_location={'cuda:0': 'cuda:2'})
    model.load_state_dict(dic)
    model.to('cuda:2')
    #model.load_state_dict(torch.load('../PretrainModel/' + 'DGCN_ModelNet40.parm'))
    #model = model.cuda()

    G_model = Generator(params)
    # G_model.load_state_dict(torch.load('../savedModel/SPU_Final8X_pointnet_att.parm'))
    # G_model.cuda()

    criterion = nn.CrossEntropyLoss()
    opt.test_losses = AverageMeter()
    infer(model,G_model, test_loader, criterion, opt)



