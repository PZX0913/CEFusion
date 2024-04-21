#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
#import pytorch_msssim #此处需要去网上下载,于5.5日新加入，用以补充损失函数，内容损失此时由梯度损失，强度损失和ssim相似性损失同时组成
import SSIM
import Diceloss
import cv2
import numpy as np
np.set_printoptions(threshold=np.sys.maxsize)

# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()
#         self.epsilon = 1e-5
#
#     def forward(self, predict, target):
#         assert predict.size() == target.size(), "the size of predict and target must be equal."
#         num = predict.size(0)
#
#         #pre = torch.sigmoid(predict).view(num, -1)#sigmod函数用来
#         pre = predict.view(num, -1)
#         tar = target.view(num, -1)#view函数用于将设置的张量铺平
#
#         intersection = (pre/255 * tar/255).sum(-1).sum()  # 利用预测值与标签相乘当作交集
#         #union = (pre + tar).sum(-1).sum()
#         union = (pre/255).sum(-1).sum() + (tar/255).sum(-1).sum()
#         # intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
#         # union = (pre + tar).sum(-1).sum()
#
#         score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
#         #score =2*(intersection+self.epsilon ) / (union+self.epsilon )
#         return score


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class NormalLoss(nn.Module):
    def __init__(self,ignore_lb=255, *args, **kwargs):
        super( NormalLoss, self).__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        return torch.mean(loss)

class Fusionloss(nn.Module):#融合损失，包括强度损失pixel以及纹理texture损失
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    # def Diceloss(self, x, y):
    #     epsilon = 1e-5
    #     #assert predict.size() == target.size(), "the size of predict and target must be equal."
    #     num1 = x.size()
    #
    #     #pre = torch.sigmoid(predict).view(num, -1)#sigmod函数用来
    #     pre = x.view(num1, -1)
    #     tar = y.view(num1, -1)#view函数用于将设置的张量铺平
    #
    #     intersection = (pre/255 * tar/255).sum(-1).sum()  # 利用预测值与标签相乘当作交集
    #     #union = (pre + tar).sum(-1).sum()
    #     union = (pre/255).sum(-1).sum() + (tar/255).sum(-1).sum()
    #     # intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
    #     # union = (pre + tar).sum(-1).sum()
    #
    #     score = 1 - 2 * (intersection + epsilon) / (union + epsilon)
    #     #score =2*(intersection+self.epsilon ) / (union+self.epsilon )
    #     return score

    def forward(self, image_vis, image_ir, labels, generate_img, i):
        image_y=image_vis[:,0:,:,:]#原式为[:,:1,:,:]
        #下面这些是于5.6日新引进的结构性ssim损失函数
        ssim_loss = SSIM.ssim#在此处引入相似性结构损失，增加图像的可视性
        ssim_loss_value = 0
        #ssim_loss_temp = ssim_loss(generate_img, image_y, normalize=True)
        #ssim_loss_temp2 = ssim_loss(generate_img, image_ir, normalize=True)
        ssim_loss_temp = ssim_loss(generate_img, image_vis, data_range=255, win=None)
        ssim_loss_temp2 = ssim_loss(generate_img, image_ir, data_range=255, win=None)
        ssim_loss_value += (1 - ssim_loss_temp)
        ssim_loss_value += (1 - ssim_loss_temp2)
        loss_ssim = ssim_loss_value

        #
        #以下这些是5.16新引进的边缘损失Dice loss函数 这里需要注意的是，输入进Canny算子边缘检测函数的变量需要为数组
        image_vis1 = image_vis.cpu().numpy()
        image_ir1 = image_ir.cpu().numpy()
        generate_img1 = generate_img.cpu().detach().numpy()
        # img3 = image_vis1[0, :, :, :]
        # img3 = np.transpose(img3, (1, 2, 0))
        #cv2.imwrite('/home/yuxuan/study/MSNet-M2SNet-main/MSNet-M2SNet-main/MSRS/Infrared/test/edge/00005N.3.png', img3*255)
        #到这里上面还是可以读到数据的，但是这里可以存取数据是因为在最后的时候对于存储的图像变量*255，那这里是不是可以理解上述数组中像素点都是为0~1,在进行unit8格式转换的时候后，就出现了错误，应该是像素点的值发生了变化
        #将图像的格式转换为unit8
        image_vis1 = image_vis1*255
        image_ir1 = image_ir1*255
        generate_img1 = generate_img1*255#为了后续的数据类型转换工作，需要将图像转换的数组乘以255
        image_vis1 = image_vis1.astype(np.uint8)
        image_ir1 = image_ir1.astype(np.uint8)
        generate_img1 = generate_img1.astype(np.uint8)
        #5.17 12.59经过验证，问题就是出在上面的转换为uint8的地方，等于将图像的值全部变为0了
        # img4 = image_vis1[0, :, :, :]
        # img4 = np.transpose(img4, (1, 2, 0))
        # cv2.imwrite('/home/yuxuan/study/MSNet-M2SNet-main/MSNet-M2SNet-main/MSRS/Infrared/test/edge/00005N.4.png', img4)#到这里，经过对数组*255的操作以后出现过曝的现象
        # print(image_vis1.shape)
        # print(image_ir1.shape)
        image_vis1 = np.transpose(image_vis1, (2, 3, 1, 0))
        image_ir1 = np.transpose(image_ir1, (2, 3, 1, 0))
        generate_img1 = np.transpose(generate_img1, (2, 3, 1, 0))
        #5.17 12.55 这里下面存储的图像也是全黑的，没有值存在，目前认为问题出在进行数据类型变换的地方-image_vis1 = image_vis1.astype(np.uint8)
        # cv2.imwrite('/home/yuxuan/study/MSNet-M2SNet-main/MSNet-M2SNet-main/MSRS/Infrared/test/edge/00005N.2.png', image_vis1[:, :, :, 0])
        # print(image_vis1.shape)
        # print(image_ir1.shape)
        # 以下这些为了正确使用Canny算子函数添加的对于输入变量的预处理过程
        # image_vis_ycrcb = cv2.cvtColor(image_vis1, cv2.COLOR_BGR2GRAY)  # 在这里需要提取出单通道出来
        # image_gen_ycrcb = cv2.cvtColor(generate_img1, cv2.COLOR_BGR2GRAY)  # 在这里需要提取出单通道出来
        # image_vis2 = image_vis_ycrcb[:, :1, :, :]  # 取全部通道
        # image_ir2 = image_ir1[:, :1, :, :]
        # generate_img2 = image_gen_ycrcb[:, :1, :, :]
        #print(image_vis1.shape)
        #5.17日上午10.52发现问题，以下变量中根本就没有值，图像是一片黑
        # image_vis1_1 = image_vis1[:, :, :, 0]
        # image_ir1_1 = image_ir1[:, :, :, 0]
        # generate_img1_1 = generate_img1[:, :, :, 0]
        image_vis1_1 = image_vis1
        image_ir1_1 = image_ir1
        generate_img1_1 = generate_img1
        # print(image_vis1_1.shape)
        # print(image_ir1_1.shape)
        #5.17目前存在的问题是提取出的边缘变量全为0，根本没有实现出对应功能
        edge = cv2.Canny(image_vis1_1, 50, 150)#这里需要注意输入进去的图像的点类型必须为unit8的类型
        edge1 = cv2.Canny(image_ir1_1, 50, 150)
        edge_result = cv2.Canny(generate_img1_1, 50, 150)
        edge_out = (edge + edge1)
        edge_result = torch.from_numpy(edge_result)
        edge_out = torch.from_numpy(edge_out)
        # print(edge_out.shape)
        # print(edge_result.shape)
        # print((edge_result).sum(-1).sum())
        # print((edge_out).sum(-1).sum())
        # predict = edge_result
        # target = edge_out

        # assert edge_result.size() == edge_out.size(), "the size of predict and target must be equal."
        # num = edge_result.size(0)
        # # pre = torch.sigmoid(predict).view(num, -1)#sigmod函数用来
        # pre = edge_result.view(num, -1)
        # tar = edge_out.view(num, -1)  # view函数用于将设置的张量铺平
        #
        # intersection = (pre / 255 * tar / 255).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        # # union = (pre + tar).sum(-1).sum()
        # union = (pre / 255).sum(-1).sum() + (tar / 255).sum(-1).sum()
        # # intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        # # union = (pre + tar).sum(-1).sum()
        #
        # loss_edge = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        # # score =2*(intersection+self.epsilon ) / (union+self.epsilon )
        diceloss = Diceloss.DiceLoss()
        loss_edge = diceloss(edge_result, edge_out)#这里输出的是张量，现在需要探究的是DiceLoss损失函数是否可以处理张量以及数组形式的变量，这样的话就不需要对得到的边缘信息进行转换处理
        #print(loss_edge.cpu().numpy())
        loss_edge = loss_edge.cpu()
        #
        x_in_max = torch.max(image_y, image_ir)#取像素中的最大值
        loss_in = F.l1_loss(x_in_max, generate_img)
        loss_pixel = loss_in
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        loss_content = loss_in+10*loss_grad+loss_ssim
        #loss_total = 0.5*(loss_in+10*loss_grad+loss_ssim)+0.5*loss_edge #更改时间为5.5日;此处15原为10，将此超参数调大是为了提高梯度对于训练的影响，使图像的背景纹理细节更加清晰，这里之所以乘以十是因为损失函数之间差了一个量级
        loss_total =(loss_in + 10 * loss_grad + 10 * loss_ssim) + 8 * loss_edge
        return loss_total, loss_pixel, loss_grad, loss_ssim, loss_content, loss_edge

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        #这里也可以从卷积核直接入手，将一个一维的单通道卷积核变为三通道
        kernelx = [[[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]],
                   [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]],
                   [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]]
        kernely = [[[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]],
                   [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]],
                   [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]]
        # kernelx = [[-1, 0, 1],
        #             [-2, 0, 2],
        #             [-1, 0, 1]]
        # kernely = [[1, 2, 1],
        #            [0, 0, 0],
        #            [-1, -2, -1]]
        #在这里进行相关处理，将一个单通道的卷积核转换为三通道，且每一个通道的参数都是一样的，保证测量损失函数一致
        #kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        #kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

if __name__ == '__main__':
    pass

