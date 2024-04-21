# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os


def prepare_data_path(dataset_path):#prepare the dataset
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))#返回指定路径下所有尾标为bmp，tif，png..等等的文件
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'test'], 'split must be "train"|"test"'#Val数据集主要的作用是用来验证

        if split == 'train':
            data_dir_vis = './MSRS/Visible/train/MSRS/'
            data_dir_ir = './MSRS/Infrared/train/MSRS/'
            data_dir_label = './MSRS/Label/train/MSRS/'
            # data_dir_vis = './RoadScene/Visible/train/RoadScene_1/'
            # data_dir_ir = './RoadScene/infrared/train/RoadScene_1/'
            # data_dir_label = './RoadScene/Label/train/RoadScene/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        if split == 'test':
            data_dir_vis = './MSRS/Visible/test/MSRS/'
            data_dir_ir = './MSRS/Infrared/test/MSRS/'
            data_dir_label = './MSRS/Label/test/MSRS/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            label_path = self.filepath_label[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 1)#需要改为1，如果是在三通道的情况下训练
            label = np.array(Image.open(label_path))
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1))/ 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32).transpose((2, 0, 1))/ 255.0
            label = np.asarray(Image.fromarray(label), dtype=np.int64)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                torch.tensor(label),
                name,
            )

        if self.split == 'test':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            label_path = self.filepath_label[index]
            image_vis = np.array(Image.open(vis_path))
            #image_inf = cv2.imread(ir_path, 0)
            image_inf = cv2.imread(ir_path, 1)#此状态是在3通道的情况下训练
            label = np.array(Image.open(label_path))
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1))/ 255.0
                #np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1))
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32).transpose((2, 0, 1)) / 255.0
            #image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32)
            #image_ir = np.expand_dims(image_ir, axis=0)
            label = np.asarray(Image.fromarray(label), dtype=np.int64)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                torch.tensor(label),
                name,
            )

        elif self.split == 'val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            image_vis = (
                #np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1))/ 255.0)
            np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(2, 0, 1))
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
            )

    def __len__(self):
        return self.length

if __name__ == '__main__':
    train_dataset = Fusion_dataset('train')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        #shuffle=True,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
        if it == 0:
            print(name)
            label.numpy()
            print(label.shape)
            image_vis.numpy()
            print(image_vis.shape)
            image_ir.numpy()
            print(image_ir.shape)
            break
