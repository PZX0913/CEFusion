import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net import res2net50_v1b_26w_4s
from torchvision import transforms
import torchvision.transforms as transforms
import torchvision
import numpy as np  
import imageio
import matplotlib.pyplot as plt  
import cv2


class CNN1(nn.Module):
    def __init__(self, channel, map_size, pad):
        super(CNN1, self).__init__()
        self.weight = nn.Parameter(torch.ones(channel, channel, map_size, map_size),
                                   requires_grad=False).cuda()  
        self.bias = nn.Parameter(torch.zeros(channel), requires_grad=False).cuda()  
        self.pad = pad  
        self.norm = nn.BatchNorm2d(channel)  
        self.relu = nn.ReLU()  

    def forward(self, x):
        out = F.conv2d(x, self.weight, self.bias, stride=1, padding=self.pad)
        out = self.norm(out)
        out = self.relu(out)
        return out


class CENet(nn.Module):
    # res2net based encoder decoder
    def __init__(self):
        super(CENet, self).__init__()
        # ResNet Backbone
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.conv_3 = CNN1(64, 3, 1)
        self.conv_5 = CNN1(64, 5, 2)

        self.x5_dem_1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.x5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))

        self.y5_dem_1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.y4_dem_1 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.y3_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.y2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.level4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.x5_dem_5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.y5_dem_5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.cov3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output64_32 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))
        self.output32_16 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16),
                                     nn.ReLU(inplace=True))
        self.output16_3 = nn.Sequential(nn.Conv2d(16, 3, kernel_size=3, padding=1))

    def forward(self, x, y):
        input = x
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x1 = self.resnet.maxpool(x)
        # ---- low-level features ----
        x2 = self.resnet.layer1(x1)
        x3 = self.resnet.layer2(x2)
        x4 = self.resnet.layer3(x3)
        x5 = self.resnet.layer4(x4)

        y = self.resnet.conv1(y)
        y = self.resnet.bn1(y)
        y = self.resnet.relu(y)
        y1 = self.resnet.maxpool(y)
        y2 = self.resnet.layer1(y1)
        y3 = self.resnet.layer2(y2)
        y4 = self.resnet.layer3(y3)
        y5 = self.resnet.layer4(y4)

        # Difference fusion
        x_y5 = x5 - y5
        y_x5 = y5 - x5
        x5_dem_1 = self.x5_dem_1(x5)
        y5_dem_1 = self.y5_dem_1(y5)
        x_y5 = self.x5_dem_1(x_y5)
        y_x5 = self.y5_dem_1(y_x5)
        x5_dem_1 = x5_dem_1 + y_x5
        y5_dem_1 = y5_dem_1 + x_y5
        x5_dem_1 = self.cov3(self.cov3(x5_dem_1) + self.cov3(y5_dem_1))

        # Difference fusion
        x_y4 = x4 - y4
        y_x4 = y4 - x4
        x4_dem_1 = self.x4_dem_1(x4)
        y4_dem_1 = self.y4_dem_1(y4)
        x_y4 = self.x4_dem_1(x_y4)
        y_x4 = self.y4_dem_1(y_x4)
        x4_dem_1 = x4_dem_1 + y_x4
        y4_dem_1 = y4_dem_1 + x_y4
        x4_dem_1 = self.cov3(self.cov3(x4_dem_1) + self.cov3(y4_dem_1))

        # Difference fusion
        x_y3 = x3 - y3
        y_x3 = y3 - x3
        x3_dem_1 = self.x3_dem_1(x3)
        y3_dem_1 = self.y3_dem_1(y3)
        x_y3 = self.x3_dem_1(x_y3)
        y_x3 = self.y3_dem_1(y_x3)
        x3_dem_1 = x3_dem_1 + y_x3
        y3_dem_1 = y3_dem_1 + x_y3
        x3_dem_1 = self.cov3(self.cov3(x3_dem_1) + self.cov3(y3_dem_1))

        # Difference fusion
        x_y2 = x2 - y2
        y_x2 = y2 - x2
        x2_dem_1 = self.x2_dem_1(x2)
        y2_dem_1 = self.y2_dem_1(y2)
        x_y2 = self.x2_dem_1(x_y2)
        y_x2 = self.y2_dem_1(y_x2)
        x2_dem_1 = x2_dem_1 + y_x2
        y2_dem_1 = y2_dem_1 + x_y2
        x2_dem_1 = self.cov3(self.cov3(x2_dem_1) + self.cov3(y2_dem_1))

        # Difference fusion
        x_y1 = x1 - y1
        y_x1 = y1 - x1
        x1_dem_1 = self.cov3(x1)
        y1_dem_1 = self.cov3(y1)
        x_y1 = self.cov3(x_y1)
        y_x1 = self.cov3(y_x1)
        x1_dem_1 = x1_dem_1 + y_x1
        y1_dem_1 = y1_dem_1 + x_y1
        x1_dem_1 = self.cov3(self.cov3(x1_dem_1) + self.cov3(y1_dem_1))

        x5_dem_1_up = F.upsample(x5_dem_1, size=x4.size()[2:],mode='bilinear')
        x5_dem_1_up_map1 = self.conv_3(x5_dem_1_up)
        x4_dem_1_map1 = self.conv_3(x4_dem_1)
        x5_dem_1_up_map2 = self.conv_5(x5_dem_1_up)
        x4_dem_1_map2 = self.conv_5(x4_dem_1)
        x5_4 = self.x5_x4(
            abs(x5_dem_1_up - x4_dem_1) + abs(x5_dem_1_up_map1 - x4_dem_1_map1) + abs(
                x5_dem_1_up_map2 - x4_dem_1_map2))


        x4_dem_1_up = F.upsample(x4_dem_1, size=x3.size()[2:], mode='bilinear')
        x4_dem_1_up_map1 = self.conv_3(x4_dem_1_up)
        x3_dem_1_map1 = self.conv_3(x3_dem_1)
        x4_dem_1_up_map2 = self.conv_5(x4_dem_1_up)
        x3_dem_1_map2 = self.conv_5(x3_dem_1)
        x4_3 = self.x4_x3(
            abs(x4_dem_1_up - x3_dem_1) + abs(x4_dem_1_up_map1 - x3_dem_1_map1) + abs(x4_dem_1_up_map2 - x3_dem_1_map2))

        x3_dem_1_up = F.upsample(x3_dem_1, size=x2.size()[2:], mode='bilinear')
        x3_dem_1_up_map1 = self.conv_3(x3_dem_1_up)
        x2_dem_1_map1 = self.conv_3(x2_dem_1)
        x3_dem_1_up_map2 = self.conv_5(x3_dem_1_up)
        x2_dem_1_map2 = self.conv_5(x2_dem_1)
        x3_2 = self.x3_x2(
            abs(x3_dem_1_up - x2_dem_1) + abs(x3_dem_1_up_map1 - x2_dem_1_map1) + abs(x3_dem_1_up_map2 - x2_dem_1_map2))

        x2_dem_1_up = F.upsample(x2_dem_1, size=x1.size()[2:], mode='bilinear')
        x2_dem_1_up_map1 = self.conv_3(x2_dem_1_up)
        x1_map1 = self.conv_3(x1)
        x2_dem_1_up_map2 = self.conv_5(x2_dem_1_up)
        x1_map2 = self.conv_5(x1)
        x2_1 = self.x2_x1(abs(x2_dem_1_up - x1) + abs(x2_dem_1_up_map1 - x1_map1) + abs(x2_dem_1_up_map2 - x1_map2))

        x5_4_up = F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear')
        x5_4_up_map1 = self.conv_3(x5_4_up)
        x4_3_map1 = self.conv_3(x4_3)
        x5_4_up_map2 = self.conv_5(x5_4_up)
        x4_3_map2 = self.conv_5(x4_3)
        x5_4_3 = self.x5_x4_x3(abs(x5_4_up - x4_3) + abs(x5_4_up_map1 - x4_3_map1) + abs(x5_4_up_map2 - x4_3_map2))

        x4_3_up = F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear')
        x4_3_up_map1 = self.conv_3(x4_3_up)
        x3_2_map1 = self.conv_3(x3_2)
        x4_3_up_map2 = self.conv_5(x4_3_up)
        x3_2_map2 = self.conv_5(x3_2)
        x4_3_2 = self.x4_x3_x2(abs(x4_3_up - x3_2) + abs(x4_3_up_map1 - x3_2_map1) + abs(x4_3_up_map2 - x3_2_map2))

        x3_2_up = F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear')
        x3_2_up_map1 = self.conv_3(x3_2_up)
        x2_1_map1 = self.conv_3(x2_1)
        x3_2_up_map2 = self.conv_5(x3_2_up)
        x2_1_map2 = self.conv_5(x2_1)
        x3_2_1 = self.x3_x2_x1(abs(x3_2_up - x2_1) + abs(x3_2_up_map1 - x2_1_map1) + abs(x3_2_up_map2 - x2_1_map2))

        x5_4_3_up = F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear')
        x5_4_3_up_map1 = self.conv_3(x5_4_3_up)
        x4_3_2_map1 = self.conv_3(x4_3_2)
        x5_4_3_up_map2 = self.conv_5(x5_4_3_up)
        x4_3_2_map2 = self.conv_5(x4_3_2)
        x5_4_3_2 = self.x5_x4_x3_x2(
            abs(x5_4_3_up - x4_3_2) + abs(x5_4_3_up_map1 - x4_3_2_map1) + abs(x5_4_3_up_map2 - x4_3_2_map2))

        x4_3_2_up = F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear')
        x4_3_2_up_map1 = self.conv_3(x4_3_2_up)
        x3_2_1_map1 = self.conv_3(x3_2_1)
        x4_3_2_up_map2 = self.conv_5(x4_3_2_up)
        x3_2_1_map2 = self.conv_5(x3_2_1)
        x4_3_2_1 = self.x4_x3_x2_x1(
            abs(x4_3_2_up - x3_2_1) + abs(x4_3_2_up_map1 - x3_2_1_map1) + abs(x4_3_2_up_map2 - x3_2_1_map2))

        x5_dem_4 = self.x5_dem_4(x5_4_3_2)
        x5_dem_4_up = F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear')
        x5_dem_4_up_map1 = self.conv_3(x5_dem_4_up)
        x4_3_2_1_map1 = self.conv_3(x4_3_2_1)
        x5_dem_4_up_map2 = self.conv_5(x5_dem_4_up)
        x4_3_2_1_map2 = self.conv_5(x4_3_2_1)
        x5_4_3_2_1 = self.x5_x4_x3_x2_x1(
            abs(x5_dem_4_up - x4_3_2_1) + abs(x5_dem_4_up_map1 - x4_3_2_1_map1) + abs(x5_dem_4_up_map2 - x4_3_2_1_map2))

        level4 = self.level4(x5_4 + x4_dem_1)
        level3 = self.level3(x4_3 + x5_4_3 + x3_dem_1)
        level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2 + x2_dem_1)
        level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1 + x1_dem_1)

        x5_dem_5 = self.x5_dem_5(x5)
        output4 = self.output4(F.upsample(x5_dem_5, size=level4.size()[2:], mode='bilinear') + level4)
        output3 = self.output3(F.upsample(output4, size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3, size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2, size=level1.size()[2:], mode='bilinear') + level1)

        output64_32 = self.output64_32(output1)
        output32_16 = self.output32_16(output64_32)
        # output64_16 = self.output64_16(output1) 
        output16_3 = self.output16_3(output32_16)
        output = F.upsample(output16_3, size=input.size()[2:], mode='bilinear')

        if self.training:
            return output
        return output


class LossNet(torch.nn.Module):
    def __init__(self, resize=True):
        super(LossNet, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target

        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.mse_loss(x, y)
        return loss
