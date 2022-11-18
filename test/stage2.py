import torch
from functions import *
from utils import *
import pandas as pd
import os


@torch.no_grad()
def stage2_(model1,dataParser,save_path,save_excel_path,num):
    #eval()状态
    model1.eval()

    losses=Averagvalue()
    loss_stage1=Averagvalue()
    #stage1的四大指标
    f1_stage1=Averagvalue()
    accuracy_stage1=Averagvalue()
    precision_stage1=Averagvalue()
    recall_stage1=Averagvalue()

    src_name_list=[]
    loss_stage1_list=[]

    #在excel中打印的
    f1_stage1_list=[]
    accuracy_stage1_list=[]
    precision_stage1_list=[]
    recall_stage1_list=[]



    for index,item in enumerate(dataParser):
        #由输入的dataParser获取输入图像名称
        _path=item['path']
        src_dir=_path['src'][0]  #在机器上训练时要取消注释
        if torch.cuda.is_available():
            type_name = src_dir.split('/')[-3]
            pred_mask_name = src_dir.split('/')[-1].replace('.jpg', '.bmp')
            type_name=type_name+'_stage1、2'
        else:
            src_dir = _path['src'][0].replace('/', '\\')
            type_name = src_dir.split('\\')[-3]
            pred_mask_name = src_dir.split('\\')[-1].replace('.jpg', '.bmp')
            src_name_list.append(pred_mask_name)
            type_name=type_name+'_stage1、2'
        pred_save_path = os.path.join(save_path, type_name)  #得到保存文件夹地址

        #输入数据
        with torch.set_grad_enabled(False):
            if torch.cuda.is_available():
                images = item['tamper_image'].cuda()
                labels_band = item['gt_band'].cuda()
                labels_dou_edge = item['gt_double_edge'].cuda()
            else:
                images=item['tamper_image']
                labels_band=item['gt_band']
                labels_dou_edge=item['gt_double_edge']

            with torch.set_grad_enabled(False):
                images.required_grad=False

                stage1_output=model1(images)

            if stage1_output[0].shape!=labels_dou_edge.shape:
                continue
            try:
                loss_stage1_value=wce_dice_huber_loss(stage1_output[0],labels_band)

                loss_stage1_list.append(loss_stage1_value.item())

                loss_stage1.update(loss_stage1_value.item())

                f1_stage1_score=my_f1_score(stage1_output[0],labels_band)
                accuracy_stage1_score=my_acc_score(stage1_output[0],labels_band)
                precision_stage1_score=my_precision_score(stage1_output[0],labels_band)
                recall_stage1_score=my_recall_score(stage1_output[0],labels_band)
                f1_stage1_list.append(f1_stage1_score)
                accuracy_stage1_list.append(accuracy_stage1_score)
                precision_stage1_list.append(precision_stage1_score)
                recall_stage1_list.append(recall_stage1_score)


                f1_stage1.update(f1_stage1_score)
                precision_stage1.update(precision_stage1_score)
                accuracy_stage1.update(accuracy_stage1_score)
                recall_stage1.update(recall_stage1_score)



                '''写入表'''
                data={
                    'srcName': src_name_list,
                    'f1_stage1': f1_stage1_list,
                    'precision_stage1': precision_stage1_list,
                    'acc_stage1': accuracy_stage1_list,
                    'recall_stage1': recall_stage1_list,
                }
                test=pd.DataFrame(data)
                test.to_excel(save_excel_path)
            except Exception as e:
                print(e)

            '''保存生成的mask'''
            '''stage1'''
            # 第一维代表的是batch_size，然后是通道数和图像尺寸，首先要进行维度顺序的转换
            pred_save_path_1 = os.path.join(pred_save_path, 'stage2')
            outputs = stage1_output[0]
            outputs = outputs.permute(0, 2, 3, 1)
            outputs_1 = outputs.cpu().detach().numpy()
            pred_mask = outputs_1[0]
            pred_mask = pred_mask.squeeze(2)
            # matplotlib.pyplot.imshow()需要数据是二维的数组或者第三维深度是3或4的三维数组
            # 这里pred_mask有三个维度，最后一层深度为1，需要用np.squeeze()压缩为二维数组
            pred_mask = pred_mask*255
            # plt.figure(1)
            # plt.imshow(pred_mask)
            # plt.show()
            pred_mask = Image.fromarray(pred_mask).convert('L')
            if os.path.exists(pred_save_path_1):
                pred_mask.save(os.path.join(pred_save_path_1, pred_mask_name))
            else:
                os.makedirs(pred_save_path_1)
                pred_mask.save(os.path.join(pred_save_path_1, pred_mask_name))



        print('[{}/{}]'.format(index,len(dataParser)))
        if index==num:
            break

    return{'loss_avg':losses.avg,
           'f1_avg_stage1':f1_stage1.avg,
           'precision_avg_stage1': precision_stage1.avg,
            'accuracy_avg_stage1': accuracy_stage1.avg,
            'recall_avg_stage1': recall_stage1.avg
           }