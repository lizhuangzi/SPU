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
from ArchiAblation.SPUfinal import Generator
from option.config import OptInit
from data.ModelNet40SPU import ModelNet40spu
import os
import pandas as pd
from option.train_option import get_train_options
from utils.xyz_util import save_xyz_file
import torch.nn.functional as F
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def infer(model,G_model, test_loader, criterion, opt):
    model.eval()
    G_model.eval()
    test_true = []
    test_pred = []
    top5 = AverageMeter()
    index = 0


    for i, (data, gt, label) in enumerate(tqdm(test_loader)):
        batch_size = data.shape[0]
        data, label = data.to(opt.device), label.to(opt.device).squeeze()
        data = data.permute(0, 2, 1)
        gt = gt.permute(0, 2, 1).cuda()

        # classification
        data.requires_grad = True
        lr_ClsResult, ff, _ = model(data)
        weight2 = torch.ones(lr_ClsResult.size()).cuda()
        # weight2 = weight2 / 40
        lrdata_jocabian, = torch.autograd.grad(lr_ClsResult, data, weight2, retain_graph=True)

        with torch.no_grad():
            preds = G_model(data, lrdata_jocabian.detach())
            preds = preds.detach()
            lr_logits, _, _ = model(data)
            logits, gbf2, _ = model(preds)
            hr_logit, hr_gbf_hl, _ = model(gt)
            # clean grident
            model.zero_grad()
            G_model.zero_grad()

        batch_size = label.size(0)
        _, pred = logits.topk(1, 1, True, True)
        _, pred2 = hr_logit.topk(1, 1, True, True)
        _, pred3 = lr_logits.topk(1,1,True,True)
        pred = pred.t()
        pred2 = pred2.t()
        pred3 = pred3.t()
        correct1 = pred.eq(label.view(1, -1).expand_as(pred))[0]
        correct2 = pred2.eq(label.view(1, -1).expand_as(pred2))[0]
        correct3 = pred3.eq(label.view(1, -1).expand_as(pred3))[0]

        arraylist = []
        for ss in range(len(correct1)):
            if correct2[ss] == False and correct1[ss] == True:
                record1 = pred[0][ss]
                record2 = pred2[0][ss]
                record3 = pred3[0][ss]

                apoints = data[ss]
                apoints = apoints.detach().cpu().numpy().T
                srpoints = preds[ss]
                srpoints = srpoints.detach().cpu().numpy().T
                gtpoints = gt[ss]
                gtpoints = gtpoints.cpu().numpy().T
                save_xyz_file(srpoints, '../SRXYZresults/%d_%d_sr.xyz' % (index,record1))
                save_xyz_file(apoints, '../SRXYZresults/%d_%d_lr.xyz' % (index,record3))
                save_xyz_file(gtpoints, '../SRXYZresults/%d_%d_hr.xyz' % (index,record2))
                index += 1

        #prec1,prec5 = accuracy(logits, label, topk=(1,5))

        # top5.update(prec5.cpu().numpy(), data.size(0))
        #
        # pred = logits.max(dim=1)[1]
        # test_true.append(label.cpu().numpy())
        # test_pred.append(pred.detach().cpu().numpy())
        #
        #
        # j=0
        # apoints = data[j]
        # apoints = apoints.detach().cpu().numpy().T
        # srpoints = preds[j]
        # srpoints = srpoints.detach().cpu().numpy().T
        # gtpoints = gt[j]
        # gtpoints = gtpoints.cpu().numpy().T
        # save_xyz_file(srpoints,'../SRXYZresults/%d_sr.xyz'%index)
        # save_xyz_file(apoints, '../SRXYZresults/%d_lr.xyz' % index)
        # save_xyz_file(gtpoints, '../SRXYZresults/%d_hr.xyz' % index)
        #
        # index+=1




    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    overall_acc = metrics.accuracy_score(test_true, test_pred)
    class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    return overall_acc, top5.avg, class_acc, opt



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

    eval_dst = ModelNet40spu(data_dir=params["dataset_dir"], partition='test', num_points=2048,upscalefactor=4)
    #eval_dst = ModelNet40_DA(data_dir=params["dataset_dir"], partition='test', num_points=1024,upscalefactor=4)
    test_loader = DataLoader(eval_dst, batch_size=50,
                                 shuffle=False, pin_memory=True, num_workers=4)

    opt.n_classes = 40

    model = PoiintNet(k=opt.n_classes,normal_channel=False)
    #model = DGCNN_cls(opt.n_classes)
    model.load_state_dict(torch.load('../PretrainModel/'+'PointNetModelNet40.parm'))
    model = model.cuda()

    G_model = Generator(params)
    G_model.load_state_dict(torch.load('../savedModel/SPU_Final4X_pointnet_att.parm'))
    G_model.cuda()

    criterion = nn.CrossEntropyLoss()
    opt.test_losses = AverageMeter()
    top1,top5, test_class_acc, opt = infer(model,G_model, test_loader, criterion, opt)
    print(top1)
    print(top5)
    print(test_class_acc)


