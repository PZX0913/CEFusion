#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
import cv2
from torch.autograd import Variable
from torchvision import transforms
from CEFusion import CENet
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from loss import OhemCELoss, Fusionloss
from optimizer import Optimizer
import torch
from torch.utils.data import DataLoader
import warnings
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')



def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            3,
            list(input_im.size())[2],
            list(input_im.size())[3],
        )
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            3,
            list(input_im.size())[2],
            list(input_im.size())[3],
        )
    )
    return out



def train_fusion(num=0, logger=None):
    lr_start = 0.001
    modelpth = './model-save'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    fusionmodel = eval('CENet')
    fusionmodel = fusionmodel().cuda()
    fusionmodel.train())
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    if logger == None:
        logger = logging.getLogger()
        setup_logger(modelpth)
    train_dataset = Fusion_dataset('train')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,   #16
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    # 
    if num>0:
        score_thres = 0.7
        ignore_idx = 255
        n_min = 8 * 640 * 480 // 8
        criteria_p = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        criteria_16 = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_fusion = Fusionloss()

    epoch = 10
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):#epoch=10
        print('\n| epo #%s begin...' % epo)
        lr_start = 0.001
        lr_decay = 0.80
        lr_this_epo = lr_start
        #lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo 

        for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
            fusionmodel = fusionmodel.cuda()
            fusionmodel.train()
            image_vis = Variable(image_vis).cuda()
            image_ir = Variable(image_ir).cuda()
            label = Variable(label).cuda()
            fusion_image = fusionmodel(image_vis, image_ir)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)

            lb = torch.squeeze(label, 1)
            optimizer.zero_grad()

            loss_fusion, loss_in, loss_grad, loss_ssim, loss_content, loss_edge = criteria_fusion(
                image_vis, image_ir, label, fusion_image, num
            )
            loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',
                        'loss_content: {loss_content:.4f}',
                        'loss_edge: {loss_edge:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=loss_ssim.item(),
                    loss_content=loss_content.item(),
                    loss_edge=loss_edge.item(),
                    time=t_intv,
                    eta=eta,
                )
            logger.info(msg)
            st = ed
    fusion_model_file = os.path.join(modelpth, 'fusion_model_add.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_file)
    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')


if __name__ == "__main__":
    time_start = time.time()  
    logpath='./logs'
    logger = logging.getLogger()
    for i in range(1):
        train_fusion(i, logger)
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
    print("training Done!")
    time_end = time.time()  
    time_sum = time_end - time_start  
    print(time_sum)
