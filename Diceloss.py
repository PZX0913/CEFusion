import torch
import numpy
import torch.nn as nn
import cv2 #导入opencv库
import torch.nn.functional as F
#numpy.set_printoptions(threshold=numpy.sys.maxsize)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        #pre = torch.sigmoid(predict).view(num, -1)#sigmod函数用来
        pre = predict.view(num, -1)
        tar = target.view(num, -1)#view函数用于将设置的张量铺平

        intersection = (pre/255 * tar/255).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        #union = (pre + tar).sum(-1).sum()
        union = (pre/255).sum(-1).sum() + (tar/255).sum(-1).sum()
        # intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        # union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        #score =2*(intersection+self.epsilon ) / (union+self.epsilon )
        return score

pre1 = cv2.imread('/home/yuxuan/study/MSNet-M2SNet-main/MSNet-M2SNet-main/MSRS/Infrared/test/edge/00004N.4.png')
tar1 = cv2.imread('/home/yuxuan/study/MSNet-M2SNet-main/MSNet-M2SNet-main/MSRS/Infrared/test/edge/00004N.4.png')

#下面这些都是diceloss函数的验证程序，经过多次实验验证，证明该函数是可行的！
# img = cv2.imread("/home/yuxuan/study/MSNet-M2SNet-main/MSNet-M2SNet-main/MSRS/Infrared/test/MSRS_24_gama.1.5/00008N.png", 1)
# img1 = cv2.imread("/home/yuxuan/study/MSNet-M2SNet-main/MSNet-M2SNet-main/MSRS/Visible/test/MSRS/00008N.png", 1)
# img2 = cv2.imread("/home/yuxuan/study/MSNet-M2SNet-main/MSNet-M2SNet-main/MSRS/Fusion/test/MSRS/00008N.png", 1)
# print(img2.shape)
# #进行canny边缘检测
# edge = cv2.Canny(img, 50, 150)
# edge1 = cv2.Canny(img1, 50, 150)
# edge_result = cv2.Canny(img2, 50, 150)
# edge_out = (edge + edge1)
#
# print(edge_result)
# print(edge_result.shape)
# #print(edge_out)
#
# pre = torch.from_numpy(edge_result)
# tar = torch.from_numpy(edge_out)
# print(pre.shape)
# loss = DiceLoss()
# predict = torch.randn(3, 4, 4)
# target = torch.randn(3, 4, 4)
#
# x = torch.from_numpy(numpy.array([[0, 1], [0.3, 0.1], [0.9, 0.6]]))
# y = torch.from_numpy(numpy.array([[0, 1], [0.3, 0.1], [0.9, 0.6]]))
#
# score = loss(predict, target)
# score1 = loss(pre, tar)
# score2 = loss(x, y)
#
# print(score)
# print(score1)
# score_1 = score1.numpy()#将张量tensor转换为数组numpy
# print(score_1)
# print(score2)