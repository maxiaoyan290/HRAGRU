import gru_data
import gru_model
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

device = 'cuda'
# 添加不同类型的节点
bert = BertModel.from_pretrained('./bert-base-Chinese')
# 参数都得改
mymodelnew = gru_model.trainmodel(87).to(device)
mymodelnew.load_state_dict(torch.load('model_state_dict_train_gru_800.pth'))

corporate_list = ['万业企业', '万方发展', '万科A', '三湘股份', '上实发展', '上海临港',
                  '上海建工', '世纪星源', '世联地产', '世联行', '世茂集团', '世荣兆业',
                  '东湖高新', '中交地产', '中体产业', '中华企业', '中南建设', '中国中铁',
                  '中国国贸', '中国宝安', '中国恒大', '中国武夷', '中国铁建', '中炬高新',
                  '丰华股份', '云南城投', '五矿地产', '京投发展', '京粮控股', '京能置业',
                  '亿城股份', '佳兆业', '保利发展', '保利地产', '信达地产', '光华控股',
                  '光明地产', '冠城大通', '创兴置业', '北京北辰实业', '北京城建', '华丽家族',
                  '华发股份', '华夏幸福', '华联控股', '华远地产', '南京高科', '南国置业',
                  '南山控股', '卧龙地产', '合肥城建', '城投控股', '外高桥',
                  '大名城', '大悦城', '大港股份', '大龙地产', '天伦置业', '天保基建', '天健集团',
                  '天地源', '天房发展', '天津松江', '宋都股份', '宝龙地产', '实达集团', '市北高新',
                  '广宇发展', '广宇集团', '广汇物流', '建发股份', '张江高科', '新湖中宝', '新黄浦',
                  '栖霞建设', '格力地产', '沙河股份', '津滨发展', '浦东金桥',
                  '海德股份', '海泰发展', '海航基础', '深振业A', '深深房A',
                  '深物业A', '渝开发', '湖南投资', '滨江集团', '珠光控股',
                  '珠江实业', '碧桂园', '福星股份', '空港股份', '粤宏远A', '粤泰股份',
                  '红星发展', '联发股份', '苏宁环球', '苏州高新',
                  '荣丰控股', '荣安地产', '荣盛发展', '莱茵体育', '蓝光发展',
                  '融创中国', '西藏城投', '财信发展', '越秀地产', '迪马股份',
                  '金地集团', '金科股份', '金融街', '金隅集团', '银亿股份',
                  '阳光股份', '陆家嘴', '雅居乐集团', '雅戈尔', '顺发恒业',
                  '首开股份', '香江控股', '高新发展']

x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
lossfuction = torch.nn.MSELoss()

for corporate in corporate_list:
    mydataset = gru_data.valDataset(corporate=corporate)

    outputs_gru = []
    labels = []
    mymodelnew.eval()
    for idx in range(len(mydataset)):
        data, label, news, policy, wobert, wognn = mydataset[idx]
        data = data.to(device)
        label = label.to(device)
        news = news.to(device)
        policy = policy.to(device)
        data = data.unsqueeze(0)
        news = news.unsqueeze(0)
        policy = policy.unsqueeze(0)

        output_gru = mymodelnew(data, news, policy)

        outputs_gru.append(output_gru.item())
        labels.append(label.item())

    x_values = range(1, len(mydataset) + 1)

    outputs = np.array(outputs_gru).reshape(1, -1)
    labels = np.array(labels).reshape(1, -1)
    condition = outputs >= 0.5
    outputs[condition] = (outputs[condition] - 0.5) * 2 + 0.5
    outputs[~condition] = 0.5 - (0.5 - outputs[~condition]) * 2

    condition = labels >= 0.5
    labels[condition] = (labels[condition] - 0.5) * 2 + 0.5
    labels[~condition] = 0.5 - (0.5 - labels[~condition]) * 2

    labels = labels.flatten().tolist()
    outputs = outputs.flatten().tolist()

    # Save results to a txt file
    with open(f'corporate_predictions/{corporate}.txt', 'w') as f:
        f.write('Predictions\tLabels\n')
        for pred, lbl in zip(outputs, labels):
            f.write(f'{pred}\t{lbl}\n')
print('done')