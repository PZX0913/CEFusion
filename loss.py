import torch
import torch.nn as nn
import torch.nn.functional as F
import SSIM
import Diceloss
import cv2
import numpy as np
np.set_printoptions(threshold=np.sys.maxsize)

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

class Fusionloss(nn.Module)
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_vis, image_ir, labels, generate_img, i):
        image_y=image_vis[:,0:,:,:]

        #SSIM loss
        ssim_loss = SSIM.ssim
        ssim_loss_value = 0
        ssim_loss_temp = ssim_loss(generate_img, image_vis, data_range=255, win=None)
        ssim_loss_temp2 = ssim_loss(generate_img, image_ir, data_range=255, win=None)
        ssim_loss_value += (1 - ssim_loss_temp)
        ssim_loss_value += (1 - ssim_loss_temp2)
        loss_ssim = ssim_loss_value

        #Edge loss
        image_vis1 = image_vis.cpu().numpy()
        image_ir1 = image_ir.cpu().numpy()
        generate_img1 = generate_img.cpu().detach().numpy()
        image_vis1 = image_vis1*255
        image_ir1 = image_ir1*255
        generate_img1 = generate_img1*255
        image_vis1 = image_vis1.astype(np.uint8)
        image_ir1 = image_ir1.astype(np.uint8)
        generate_img1 = generate_img1.astype(np.uint8)
        image_vis1 = np.transpose(image_vis1, (2, 3, 1, 0))
        image_ir1 = np.transpose(image_ir1, (2, 3, 1, 0))
        generate_img1 = np.transpose(generate_img1, (2, 3, 1, 0))
        image_vis1_1 = image_vis1
        image_ir1_1 = image_ir1
        generate_img1_1 = generate_img1
        edge = cv2.Canny(image_vis1_1, 50, 150)
        edge1 = cv2.Canny(image_ir1_1, 50, 150)
        edge_result = cv2.Canny(generate_img1_1, 50, 150)
        edge_out = (edge + edge1)
        edge_result = torch.from_numpy(edge_result)
        edge_out = torch.from_numpy(edge_out)

        diceloss = Diceloss.DiceLoss()
        loss_edge = diceloss(edge_result, edge_out)
        loss_edge = loss_edge.cpu()

        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)
        loss_pixel = loss_in
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        loss_content = loss_in+10*loss_grad+10*loss_ssim

        loss_total =(loss_in + 10 * loss_grad + 10 * loss_ssim) + 8 * loss_edge
        return loss_total, loss_pixel, loss_grad, loss_ssim, loss_content, loss_edge

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
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

