from os.path import isfile

import torch
import numpy as np
import os, sys

sys.path.append('../')
sys.path.append('./')
import torchvision.transforms
import matplotlib.pyplot as plt
import argparse
from dataloader import *
from stage1 import stage1_
from stage2 import stage2_
import pandas as pd

from model.one_unet_v2 import UNetStage as UNet
from model.one_unet_v2 import Cat_Unet as Cat
from stage3 import *

parser = argparse.ArgumentParser(description='Save_path, test_src and test_gt, etc.')
parser.add_argument('--stage1', default='', help='stage1 parameters')
parser.add_argument('--stage2', default='', help='stage2 parameters')
parser.add_argument('--stage3', default='', help='stage3 parameters')
parser.add_argument('--test_src_folder', default='',
                    help='The folder of test images.')
parser.add_argument('--test_gt_folder', default='',
                    help='The folder of corresponding splicing trace ground truth.')
parser.add_argument('--save_dir', default='', help='save path')
parser.add_argument('--num', default=2000, help='Maximum number of tests at a time.')
parser.add_argument('--folder_name', default='PDB_Net_results', help='保存文件夹名称')
args = parser.parse_args()


# Our model takes a lot of computing resources, and the image size used for training is (320,320,3).
# Therefore, when testing images with high resolution, we suggest only testing the result of stage1 branch.
# Please do not resize the images, as this may cause the model to be unrecognized.

class paralleling_compareing_method:
    def __init__(self, using_data=None):
        save_path = os.path.join(args.save_dir, args.folder_name)
        self.save_path = save_path
        self.using_data = using_data
        testData = TamperDataset(test_src_folder=args.test_src_folder, test_gt_folder=args.test_gt_folder)
        self.testDataLoader = torch.utils.data.DataLoader(testData, batch_size=1, num_workers=0)
        name = args.test_src_folder.split('/')[-1]
        name2 = args.test_src_folder.split('\\')[-1]
        # item is the name of test_src_folder
        if len(str(name)) > len(str(name2)):
            self.item = name2
        else:
            self.item = name

    def pred_test_stage1(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            pass
        str1 = self.item + '_stage1_每个样本的指标信息.xlsx'
        self.excel_save_path_1 = os.path.join(self.save_path, str1)
        str2 = str(self.item) + '_stage1_总体指标信息.xlsx'
        self.excel_save_path_2 = os.path.join(self.save_path, str2)

        model1 = UNet()
        if torch.cuda.is_available():
            model1.cuda()
            if isfile(args.stage1):
                checkpoint1 = torch.load(args.stage1)
                model1.load_state_dict(checkpoint1['state_dict'])
                print('加载成功！')
            else:
                print('错误类型：模型参数地址不存在！')
                sys.exit()
        else:
            model1.cpu()
            if isfile(args.stage1):
                checkpoint1 = torch.load(args.stage1, map_location=torch.device('cpu'))
                model1.load_state_dict(checkpoint1['state_dict'])
                print('加载成功！')
            else:
                print('错误类型：模型参数地址不存在！')
                sys.exit()

        # 传参调用统一的方法，参数：三个model,dataParser,mask_save_path,excel_save_path_1
        result = stage1_(model1, self.testDataLoader, save_path=self.save_path,
                         save_excel_path=self.excel_save_path_1, num=args.num)

        data = {
            'f1_stage1': result['f1_avg_stage1'],
            'precision_stage1': result['precision_avg_stage1'],
            'accuracy_stage1': result['accuracy_avg_stage1'],
            'recall_stage1': result['recall_avg_stage1']
        }
        test = pd.DataFrame(data, index=[0])
        test.to_excel(self.excel_save_path_2)

    def pred_test_stage2(self):
        # self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            pass
        str1 = self.item + '_stage2_每个样本的指标信息.xlsx'
        self.excel_save_path_1 = os.path.join(self.save_path, str1)
        str2 = str(self.item) + '_stage2_总体指标信息.xlsx'
        self.excel_save_path_2 = os.path.join(self.save_path, str2)
        model2 = UNet()
        if torch.cuda.is_available():
            model2.cuda()
            if isfile(args.stage2):
                checkpoint1 = torch.load(args.stage2)
                model2.load_state_dict(checkpoint1['state_dict'])
                print('加载成功！')
            else:
                print('错误类型：模型参数地址不存在！')
                sys.exit()
        else:
            model2.cpu()
            if isfile(args.stage1):
                checkpoint1 = torch.load(args.stage2, map_location=torch.device('cpu'))
                model2.load_state_dict(checkpoint1['state_dict'])
                print('加载成功！')
            else:
                print('错误类型：模型参数地址不存在！')
                sys.exit()

        # 传参调用统一的方法，参数：三个model,dataParser,mask_save_path,excel_save_path_1
        result = stage2_(model2, self.testDataLoader, save_path=self.save_path,
                         save_excel_path=self.excel_save_path_1, num=args.num)

        data = {
            'f1_stage2': result['f1_avg_stage1'],
            'precision_stage2': result['precision_avg_stage1'],
            'accuracy_stage2': result['accuracy_avg_stage1'],
            'recall_stage2': result['recall_avg_stage1']
        }
        test = pd.DataFrame(data, index=[0])
        test.to_excel(self.excel_save_path_2)

    def pred_test_stage3(self):
        # self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            pass
        str1 = self.item + '_stage3_每个样本的指标信息.xlsx'
        self.excel_save_path_1 = os.path.join(self.save_path, str1)
        str2 = str(self.item) + '_stage3_总体指标信息.xlsx'
        self.excel_save_path_2 = os.path.join(self.save_path, str2)
        model1 = UNet()
        model2 = UNet()
        model3 = Cat()
        if torch.cuda.is_available():
            model1.cuda()
            model2.cuda()
            model3.cuda()
            if not isfile(args.stage3):
                print('None')
            if isfile(args.stage3) and isfile(args.stage1) and isfile(args.stage2):
                checkpoint1 = torch.load(args.stage1)
                checkpoint2 = torch.load(args.stage2)
                checkpoint3 = torch.load(args.stage3)
                model1.load_state_dict(checkpoint1['state_dict'])
                model2.load_state_dict(checkpoint2['state_dict'])
                model3.load_state_dict(checkpoint3['state_dict'])
                print('加载成功！')
            else:
                print('错误类型：模型参数地址不存在！')
                sys.exit()
        else:
            model3.cpu()
            if isfile(args.stage3) and isfile(args.stage1) and isfile(args.stage2):
                checkpoint1 = torch.load(args.stage1, map_location=torch.device('cpu'))
                checkpoint2 = torch.load(args.stage2, map_location=torch.device('cpu'))
                checkpoint3 = torch.load(args.stage3, map_location=torch.device('cpu'))
                model1.load_state_dict(checkpoint1['state_dict'])
                model2.load_state_dict(checkpoint2['state_dict'])
                model3.load_state_dict(checkpoint3['state_dict'])
                print('加载成功！')
            else:
                print('错误类型：模型参数地址不存在！')
                sys.exit()

        result = stage3_result(model1, model2, model3, self.testDataLoader, save_path=self.save_path,
                               save_excel_path=self.excel_save_path_1, num=args.num)

        try:
            data = {
                'f1_stage3': result['f1_avg_stage3'],
                'precision_stage3': result['precision_avg_stage3'],
                'accuracy_stage3': result['accuracy_avg_stage3'],
                'recall_stage3': result['recall_avg_stage3']
            }
            test = pd.DataFrame(data, index=[0])
            test.to_excel(self.excel_save_path_2)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    # Please ensure the args.test_src_folder and args.test_gt_folder are not empty.
    # And if test not on images with high resolution (like images in In-The-Wild), please use scheme 1 to test on these images.

    if not os.path.exists(args.test_src_folder) or not os.path.exists(args.test_gt_folder):
        print('Please check the test folder, they do not exist!')
        sys.exit(0)
    # paralleling_compareing_method().pred_test_stage1() # scheme 1
    # paralleling_compareing_method().pred_test_stage2() # scheme 2
    paralleling_compareing_method().pred_test_stage3()  # scheme 3
