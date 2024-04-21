#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
#np.set_printoptions(threshold=np.sys.maxsize)
import cv2
from torch.autograd import Variable
from torchvision import transforms
#from FusionNet import FusionNet 这里进行修改，从自己的函数中引入设计好的差分融合网络
from CEFusion import CENet
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
# from model_TII import BiSeNet
# from cityscapes import CityScapes
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
    # num: control the segmodel
    lr_start = 0.001#用来设置初始的学习率
    modelpth = './model-save'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)#构建存放的文件夹
    fusionmodel = eval('CENet')#(output=1),不启用BatchNormalization 和 Dropout，保证BN和dropout不发生变化，pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果。
    fusionmodel = fusionmodel().cuda()#调用GPU进行训练
    #fusionmodel().train()
    fusionmodel.train()#设置训练,启用 Batch Normalization 和 Dropout,model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    #optimizer = torch.optim.Adam(fusionmodel().parameters(), lr=lr_start)
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    if logger == None:
        logger = logging.getLogger()#日志对象
        setup_logger(modelpth)
    train_dataset = Fusion_dataset('train')#读入在数据集中设置好的用于训练的数据集
    print("the training dataset is length:{}".format(train_dataset.length))#输出训练数据集的图像总数
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )#加载用于训练的数据集
    train_loader.n_iter = len(train_loader)#表示将训练集一共分为多少份？一般值为图像总数/batchsize
    # 
    if num>0:
        score_thres = 0.7
        ignore_idx = 255
        n_min = 8 * 640 * 480 // 8
        criteria_p = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        criteria_16 = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_fusion = Fusionloss()#将融合损失变量赋予criteria_fusion

    epoch = 10
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):#epoch=10
        print('\n| epo #%s begin...' % epo)
        lr_start = 0.001#开始的学习率
        lr_decay = 0.80#学习率衰减
        #lr_this_epo = lr_start * lr_decay ** (epo - 1)#学习率动态变化，随时迭代次数的改变，学习率也发生相应的变化
        lr_this_epo = lr_start
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo #学习率发生改变

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
            ed = time.time()#记录实验开始的时间
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
                        #'loss_seg: {loss_seg:.4f}',
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
                    #loss_seg=loss_seg,
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
    time_start = time.time()  # 记录开始时间
    # function()   执行的程序
    # argparse模块是Python内置的一个用于命令项选项与参数解析的模块，argparse模块可以让人轻松编写用户友好的命令行接口。通过在程序中定义好我们需要的参数，然后argparse将会从sys.argv
    # 解析出这些参数。argparse模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='CENet-Fusion')#设置该网络模型的名称
    parser.add_argument('--batch_size', '-B', type=int, default=16)#设置训练的批次为16
    parser.add_argument('--gpu', '-G', type=int, default=0)#调用GPU参与训练
    parser.add_argument('--num_workers', '-j', type=int, default=8)#num_workers代表着在训练过程中的工作进程
    args = parser.parse_args()
    logpath='./logs'
    logger = logging.getLogger()#调出整个训练过程的日志
    for i in range(1):
        train_fusion(i, logger)
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
    print("training Done!")
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)