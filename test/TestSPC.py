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
from option.config import OptInit
from data.ModelNet40SPU import ModelNet40spu
from data.ModelNet40 import ModelNet40
import os
import pandas as pd
from option.train_option import get_train_options
import random
from loss.loss import Loss
from torch.nn.functional import cosine_similarity,pairwise_distance

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def infer(model, test_loader, criterion, opt):
    model.eval()
    test_true = []
    test_pred = []
    top5 = AverageMeter()
    cd_list = []
    emd_list = []
    hd_list = []
    enclin_result = 0
    Loss_fn = Loss()
    total_number = 0

    with torch.no_grad():
        for i, (data, gt, label) in enumerate(tqdm(test_loader)):
            batch_size = data.shape[0]
            data, label = data.to(opt.device), label.to(opt.device).squeeze()
            data = data.permute(0, 2, 1)
            gt = gt.permute(0, 2, 1).cuda()

            logits,gbf,_ = model(data)
            hr_logit, hr_gbf_hl, _ = model(gt)

            prec1,prec5 = accuracy(logits, label, topk=(1,5))
            top5.update(prec5.cpu().numpy(), data.size(0))

            pred = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(pred.detach().cpu().numpy())

            data = data.permute(0, 2, 1).contiguous()
            gt = gt.permute(0, 2, 1)
            cd = Loss_fn.get_cd_loss2(data, gt, 1.0)
            cd_list.append(cd.item())
            hd = Loss_fn.get_hd_loss(data, gt, 1.0)
            hd_list.append(hd.item())
            emd = Loss_fn.get_emd_loss(data, gt, 1.0)
            emd_list.append(emd.item())

            enclin_dist = pairwise_distance(hr_gbf_hl, gbf)
            enclin_result += enclin_dist.sum().cpu().item()

            total_number += batch_size

        meancd = np.mean(cd_list)
        meanemd = np.mean(emd_list)
        meanhd = np.mean(hd_list)
        meanfenu = enclin_result / total_number
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        overall_acc = metrics.accuracy_score(test_true, test_pred)
        class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    return overall_acc, top5.avg, class_acc,meancd,meanemd,meanhd,meanfenu, opt



if __name__ == '__main__':
    opt = OptInit()._get_args()
    params = get_train_options()

    seed = 1234
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

    #train_loader = DataLoader(ModelNet40(data_dir=opt.data_dir, partition='train', num_points=opt.num_points),
    #                          num_workers=8, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    #test_loader = DataLoader(ModelNet40(data_dir=opt.data_dir, partition='test', num_points=opt.num_points),
    #                         num_workers=4, batch_size=opt.test_batch_size, shuffle=True, drop_last=False)

    eval_dst = ModelNet40spu(data_dir=params["dataset_dir"], partition='test', num_points=1024,upscalefactor=16)
    #eval_dst = ModelNet40(data_dir=params["dataset_dir"], partition='test')
    test_loader = DataLoader(eval_dst, batch_size=opt.test_batch_size,
                                 shuffle=False, pin_memory=True, num_workers=4)

    opt.n_classes = 40

    # model = PoiintNet(k=opt.n_classes,normal_channel=False)
    # model.load_state_dict(torch.load('../PretrainModel/'+'PointNetModelNet40.parm'))
    # model = model.cuda()
    model = DGCNN_cls(opt.n_classes)
    model.load_state_dict(torch.load('../PretrainModel/'+'DGCN_ModelNet40.parm'))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    opt.test_losses = AverageMeter()
    top1,top5, test_class_acc,meancd, meanemd,meanhd,fd,opt = infer(model, test_loader, criterion, opt)
    print("Top1 acc: %f"%top1)
    print("Top5 acc: %f"%top5)
    print("Cls ave %f"%test_class_acc)
    print("CD: %f" % meancd)
    print("Emd: %f" % meanemd)
    print("HD: %f"%meanhd)
    print("F-Euc: %f"%fd)

