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
from ClsNetwork.DGCNNcls import DGCNN_cls
from option.config import OptInit
from data.ModelNet40 import ModelNet40,ModelNet40_DA,ModelNet40_LR
from data.ModelNet40SPU import ModelNet40spu
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def train(model, train_loader, test_loader, opt):
    logging.info('===> Init the optimizer ...')
    criterion = nn.CrossEntropyLoss()

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=opt.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    logging.info('===> Init Metric ...')
    opt.train_losses = AverageMeter()
    opt.test_losses = AverageMeter()
    best_test_overall_acc = 0.
    avg_acc_when_best = 0.
    globalepoch = 0

    ValEXPresult = {'Top1acc': [],'Top5acc': [],'Clsacc':[]}

    logging.info('===> start training ...')
    for e in range(opt.epochs):
        globalepoch+=1
        # reset tracker
        opt.train_losses.reset()
        opt.test_losses.reset()

        train_overall_acc, train_class_acc, opt = train_step(model, train_loader, optimizer, criterion, opt,e)
        test_overall_acc_top1,test_overall_acc_top5, test_class_acc, opt = infer(model, test_loader, criterion, opt)

        ValEXPresult['Top1acc'].append(test_overall_acc_top1)
        ValEXPresult['Top5acc'].append(test_overall_acc_top5)
        ValEXPresult['Clsacc'].append(test_class_acc)

        scheduler.step()
        if best_test_overall_acc<test_overall_acc_top1:
            best_test_overall_acc = test_overall_acc_top1
            torch.save(model.state_dict(), '../savedModel/DGCN_ModelNet40cls_1024.parm')

        out_path = '../statistics/'
        data_frame = pd.DataFrame(
            data={'Top1acc': ValEXPresult['Top1acc'], 'Top5acc': ValEXPresult['Top5acc'], 'Clsacc': ValEXPresult['Clsacc']},
            index=range(1, globalepoch + 1))
        data_frame.to_csv(out_path + 'DGCN_ModelNet40cls_1024' + '.csv', index_label='Epoch')

import random
def train_step(model, train_loader, optimizer, criterion, opt,epoch):
    model.train()

    train_pred = []
    train_true = []
    tqdmtrain = tqdm(train_loader)
    for data, _,label in tqdmtrain:
        batch,c,dim = data.size()
        data, label = data.to(opt.device), label.to(opt.device).squeeze()
        data = data.permute(0, 2, 1)

        optimizer.zero_grad()
        #######DATrain##########
        # a = random.randint(0, 1)
        # n_point = 1024
        # if a == 0:
        #     n_point = 1024
        # else:
        #     n_point = 1024 // 4
        # data = data[:,:,0:n_point]
        ##########DATrain#########
        logits,_,_ = model(data)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        opt.train_losses.update(loss.item())

        preds = logits.max(dim=1)[1]
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        tqdmtrain.set_description(desc='epoch%d loss:%f' % (epoch,(loss / batch).item()))

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    overall_acc = metrics.accuracy_score(train_true, train_pred)
    class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
    return overall_acc, class_acc, opt


def infer(model, test_loader, criterion, opt):
    model.eval()
    test_true = []
    test_pred = []
    top5 = AverageMeter()

    with torch.no_grad():
        for i, (data, _,label) in enumerate(tqdm(test_loader)):
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


def save_ckpt(model, optimizer, scheduler, opt, name_post):
    # ------------------ save ckpt
    filename = '{}/{}_model.pth'.format(opt.ckpt_dir, opt.jobname + '-' + name_post)
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    state = {
        'epoch': opt.epoch,
        'state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_value': opt.best_value,
    }
    torch.save(state, filename)
    logging.info('save a new best model into {}'.format(filename))

#1. point number
#2. dataset
#3. save_file_name
if __name__ == '__main__':
    opt = OptInit()._get_args()
    logging.info('===> Creating dataloader ...')

    # train_loader = DataLoader(ModelNet40spu(data_dir=opt.data_dir, partition='train', num_points=1024,upscalefactor=16),
    #                           num_workers=8, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(ModelNet40spu(data_dir=opt.data_dir, partition='test', num_points=1024,upscalefactor=16),
    #                          num_workers=4, batch_size=opt.test_batch_size, shuffle=False, drop_last=False)

    train_loader = DataLoader(ModelNet40(num_points=1024,data_dir=opt.data_dir, partition='train'),
                              num_workers=4, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(num_points=1024,data_dir=opt.data_dir, partition='test'),
                             num_workers=4, batch_size=opt.test_batch_size, shuffle=False, drop_last=False)

    opt.n_classes = 40

    logging.info('===> Loading the network ...')
    model = DGCNN_cls(opt.n_classes)
    model = model.cuda()


    if opt.phase == 'train':
        train(model, train_loader, test_loader, opt)

    else:
        criterion = nn.CrossEntropyLoss()
        opt.test_losses = AverageMeter()
        test_overall_acc, test_class_acc, opt = infer(model, test_loader, criterion, opt)
        logging.info(
            'Test Overall Acc {:.4f}, Its test avg acc {:.4f}.'.
                format(test_overall_acc, test_class_acc))
