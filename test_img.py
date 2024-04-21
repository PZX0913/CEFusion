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

def run_fusion(type='test'):  
    fusion_model_path = './model/fusion_model_22.pth'
    fused_dir = os.path.join('./MSRS/Fusion', type, 'MSRS-results')
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = eval('CENet')
    fusionmodel = fusionmodel().cuda() 
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('done!')
    test_dataset = Fusion_dataset(type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            labels = Variable(labels)
            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
            fusion_image = fusionmodel(images_vis, images_ir)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            save_img_path = './MSRS/Visible/test/MSRS/'
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image_rgb = cv2.imread(save_img_path + name[k], 1)
                # gray
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                img2 = np.zeros_like(image)
                img2[:, :, 0] = gray
                img2[:, :, 1] = gray
                img2[:, :, 2] = gray
                image = img2
                ycrcb_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCR_CB)
                img3 = np.zeros_like(image)
                img3[:, :, 0] = image[:, :, 0]
                img3[:, :, 1] = ycrcb_image[:, :, 1]
                img3[:, :, 2] = ycrcb_image[:, :, 2]
                img4 = cv2.cvtColor(img3, cv2.COLOR_YCR_CB2RGB)
                save_path = os.path.join(fused_dir, name[k])
                cv2.imwrite(save_path, img4)
                print('Fusion {0} Sucessfully!'.format(save_path))


if __name__ == "__main__":
    time_start = time.time()  # start time
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='CENet-Fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=8)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    run_fusion('test')
    print("Fusion Image Sucessfully!")
    print("Test Done!")
    time_end = time.time()  # end time
    time_sum = time_end - time_start  # total
    print(time_sum)
