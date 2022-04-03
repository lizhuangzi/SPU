import numpy as np
import logging
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

import sklearn.metrics as metrics
from tqdm import tqdm
from Clsutils.eval import accuracy
from Clsutils.misc import AverageMeter
from ClsNetwork.DGCNNcls import DGCNN_cls
from option.config import OptInit
from data.ModelNet40SPU import ModelNet40spu,ModelNet40_DT
from data.ModelNet40 import ModelNet40
import os
import pandas as pd
from option.train_option import get_train_options
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def infer(model, test_loader, criterion, opt):
    model.eval()
    test_true = []
    test_pred = []
    top5 = AverageMeter()

    with torch.no_grad():
        for i, (data,_, label) in enumerate(tqdm(test_loader)):
            data, label = data.to(opt.device), label.to(opt.device).squeeze()
            data = data.permute(0, 2, 1)

            logits,_,_ = model(data)
            loss = criterion(logits, label.squeeze())

            prec1,prec5 = accuracy(logits, label, topk=(1,5))
            top5.update(prec5.cpu().numpy(), data.size(0))

            pred = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(pred.detach().cpu().numpy())

            opt.test_losses.update(loss.item())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        overall_acc = metrics.accuracy_score(test_true, test_pred)
        class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    return overall_acc, top5.avg, class_acc, opt



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

    eval_dst = ModelNet40spu(data_dir=params["dataset_dir"], partition='test', num_points=1024,upscalefactor=8)
    #eval_dst = ModelNet40(data_dir=params["dataset_dir"],partition='test')
    test_loader = DataLoader(eval_dst, batch_size=16,
                                 shuffle=False, pin_memory=True, num_workers=4)

    opt.n_classes = 40

    model = DGCNN_cls(opt.n_classes)
    #model.cuda(device=3)
    A= torch.load('../savedModel/' + 'DGCN_ModelNet40cls_fitLR8x.parm')
    model.load_state_dict(A)
    model.cuda()


    criterion = nn.CrossEntropyLoss()
    opt.test_losses = AverageMeter()
    top1,top5, test_class_acc, opt = infer(model, test_loader, criterion, opt)
    print(top1)
    print(top5)
    print(test_class_acc)


