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
from ArchiAblation.SPUfinal16x import Generator
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
from loss.loss import Loss
from torch.nn.functional import cosine_similarity,pairwise_distance
from chamfer_distance import chamfer_distance
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def infer(model, G_model, test_loader, criterion, opt):
    model.eval()
    G_model.eval()
    test_true = []
    test_pred = []
    cd_list = []
    hd_list=[]
    emd_list = []
    cosresult = 0
    enclin_result = 0
    top1 = AverageMeter()
    top5 = AverageMeter()
    index = 0
    total_number = 0
    Loss_fn = Loss()

    for i, (data, gt, label) in enumerate(tqdm(test_loader)):
        batch_size = data.shape[0]
        data, label = data.to(opt.device), label.to(opt.device).squeeze()
        data = data.permute(0, 2, 1)
        gt = gt.permute(0, 2, 1).cuda()

        # classification
        data.requires_grad = True
        lr_ClsResult, ff, _ = model(data)
        weight2 = torch.ones(lr_ClsResult.size()).cuda()
        weight2 = weight2 *2
        lrdata_jocabian, = torch.autograd.grad(lr_ClsResult, data, weight2, retain_graph=True)

        with torch.no_grad():
            preds = G_model(data, lrdata_jocabian.detach())
            preds = preds.detach()
            generateClsResult, gbf2, _ = model(preds)
            hr_logit, hr_gbf_hl, _ = model(gt)
            # clean grident
            model.zero_grad()
            G_model.zero_grad()


        prec1, prec5 = accuracy(generateClsResult, label, topk=(1, 5))
        top1.update(prec1.cpu().numpy(), data.size(0))
        top5.update(prec5.cpu().numpy(), data.size(0))

        pred = generateClsResult.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(pred.detach().cpu().numpy())

        preds = preds.permute(0, 2, 1).contiguous()
        gt = gt.permute(0, 2, 1)
        #data = F.upsample_nearest(data, scale_factor=4)
        data = data.permute(0,2,1)
        emd = Loss_fn.get_emd_loss(preds, gt, 1.0)
        cd = Loss_fn.get_cd_loss2(preds, gt, 1.0)
        emd_list.append(emd.item())
        cd_list.append(cd.item())
        hd = Loss_fn.get_hd_loss(preds, gt, 1.0)
        hd_list.append(hd.item())

        cossim = cosine_similarity(hr_gbf_hl, gbf2)
        cosresult += cossim.sum().cpu().item()
        enclin_dist = pairwise_distance(hr_gbf_hl, gbf2)
        enclin_result += enclin_dist.sum().cpu().item()

        # save xyz
        total_number += batch_size
        # j=2
        # apoints = data[j]
        # apoints = apoints.cpu().numpy().T
        # srpoints =  preds[j]
        # srpoints = srpoints.cpu().numpy().T
        # gtpoints = gt[j]
        # gtpoints = gtpoints.cpu().numpy()
        # save_xyz_file(srpoints,'../SRXYZresults/%d_sr.xyz'%index)
        # save_xyz_file(apoints, '../SRXYZresults/%d_lr.xyz' % index)
        # save_xyz_file(gtpoints, '../SRXYZresults/%d_hr.xyz' % index)
        # save_ply('../SRXYZresults/%d_sr.ply'%index,srpoints)

    meanemd = np.mean(emd_list)
    meancd = np.mean(cd_list)
    meanhd = np.mean(hd_list)
    meanfenu = enclin_result / total_number
    meanfcos = cosresult / total_number
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    overall_acc = metrics.accuracy_score(test_true, test_pred)
    class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    return overall_acc, top5.avg, class_acc, meanemd, meancd,meanhd, meanfenu, meanfcos, opt



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

    eval_dst = ModelNet40spu(data_dir=params["dataset_dir"], partition='test', num_points=1024,upscalefactor=16)
    #eval_dst = ModelNet40_DA(data_dir=params["dataset_dir"], partition='test', num_points=1024,upscalefactor=4)
    test_loader = DataLoader(eval_dst, batch_size=50,
                                 shuffle=False, pin_memory=True, num_workers=4)

    opt.n_classes = 40

    model = PoiintNet(k=opt.n_classes,normal_channel=False)
    #model = DGCNN_cls(opt.n_classes)
    model.load_state_dict(torch.load('../PretrainModel/'+'PointNetModelNet40.parm'))
    #model.load_state_dict(torch.load('../PretrainModel/'+'DGCN_ModelNet40.parm'))
    model = model.cuda()

    G_model = Generator(params)
    print('# model parameters:', sum(param.numel() for param in G_model.parameters())/1000000)
    G_model.load_state_dict(torch.load('../Ablation16x/statistics/SPU_Final16X_wcheng2.parm'))
    G_model.cuda()

    criterion = nn.CrossEntropyLoss()
    opt.test_losses = AverageMeter()
    top1,top5, test_class_acc,meanemd,meancd,meanhd, meanfenu,meanfcos,opt = infer(model,G_model, test_loader, criterion, opt)

    print("Top1 acc: %f"%top1)
    print("Top5 acc: %f"%top5)
    print("Cls ave %f"%test_class_acc)
    print("Emd: %f"%meanemd)
    print("CD: %f"%meancd)
    print("HD: %f" % meanhd)
    print("F-Euc: %f"%meanfenu)
    print("F-Cos: %f"%meanfcos)


