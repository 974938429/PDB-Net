import sys
import traceback
import time
import random

import torch
from rich.progress import track
from torch.utils.data import Dataset,DataLoader
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import rich
import numpy as np
from PIL import ImageFilter
from torchvision.transforms import transforms
import cv2 as cv
import matplotlib.pyplot as plt

class TamperDataset(Dataset):
    def __init__(self,test_src_folder,test_gt_folder):
        self.test_src_list, self.test_gt_list = [], []
        if not os.path.exists(test_src_folder) or not os.path.exists(test_gt_folder):
            print('Please check the test folder, they do not exist!')
            sys.exit(0)
        for idx,src_name in enumerate(os.listdir(test_src_folder)):
            # By default, the name of the original image is the same as that of the corresponding truth value.
            # Please modify it according to the actual situation
            gt_name = src_name.replace('jpg','png')         # You need change here according to different datasets.
            src_img_path = os.path.join(test_src_folder,src_name)
            gt_img_path = os.path.join(test_gt_folder,gt_name)
            src_img = Image.open(src_img_path).convert('RGB')
            gt_img = Image.open(gt_img_path).convert('L')  # Please check gt_img is the splicing trace ground truth.
            src_img,gt_img = np.array(src_img),np.array(gt_img)
            if src_img.shape[0] == gt_img.shape[0] and src_img.shape[1] == gt_img.shape[1]:
                self.test_src_list.append(src_img_path)
                self.test_gt_list.append(gt_img_path)

    def __getitem__(self,index):
        tamper_path=self.test_src_list[index]
        gt_path=self.test_gt_list[index]
        try:
            img=Image.open(tamper_path).convert('RGB')
            gt=Image.open(gt_path).convert('RGB')
        except Exception as e:
            traceback.print_exc(e)
        # To generate the tampered band or double edges.
        try:
            #将255和100都换为1
            gt_band=self.__gen_band(gt)
            gt_dou_edge = self.__to_dou_edge(gt)
        except Exception as e:
            print(e)

        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
        ])(img)
        gt_band = transforms.ToTensor()(gt_band)
        gt_dou_edge = transforms.ToTensor()(gt_dou_edge)
        sample={'tamper_image':img,'gt_band':gt_band,'gt_double_edge':gt_dou_edge,
                'path':{'src':tamper_path,'gt':gt_path}}
        return sample

    def __len__(self):
        length=len(self.test_src_list)
        return length

    def __gen_band(self,gt,dilate_window=5):
        _gt=gt.copy()
        if len(_gt.split())==3:
            _gt=_gt.split()[0]
        else:
            pass
        _gt=np.array(_gt,dtype='uint8')
        if max(_gt.reshape(-1))==255:
            _gt=np.where((_gt==255)|(_gt==100),1,0)
            _gt=np.array(_gt,dtype='uint8')
        else:
            pass
        _gt=cv.merge([_gt])#对拆封的通道合并
        kernel=np.ones((dilate_window,dilate_window),np.uint8)
        _band=cv.dilate(_gt,kernel)
        _band=np.array(_band,dtype='uint8')
        _band=np.where(_band==1,255,0)
        _band=Image.fromarray(np.array(_band,dtype='uint8'))
        if len(_band.split())==3:
            _band=np.array(_band)[:,:,0]
        else:
            _band=np.array(_band)
        return _band

    def __to_dou_edge(self,dou_em):
        #转化100为255为边缘
        _dou_em = dou_em.copy()
        if len(_dou_em.split()) == 3:
            _dou_em = _dou_em.split()[0]
        else:
            pass
        _dou_em=np.array(_dou_em)
        _dou_em=np.where(_dou_em==100,255,_dou_em)
        _dou_em=np.where(_dou_em==50,0,_dou_em)
        _dou_em = Image.fromarray(np.array(_dou_em, dtype='uint8'))
        if len(_dou_em.split()) == 3:
            _band = np.array(_dou_em)[:, :, 0]
        else:
            _dou_em= np.array(_dou_em)
        return _dou_em


if __name__=='__main__':
    print('start')
    mytestdataset = TamperDataset(test_src_folder='',test_gt_folder='')
    dataloader=torch.utils.data.DataLoader(mytestdataset,batch_size=1,num_workers=1)
    print(len(mytestdataset))



