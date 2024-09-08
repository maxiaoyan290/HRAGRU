import gru_data
import gru_model
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertModel,BertTokenizer
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

device = 'cuda:1'
# 添加不同类型的节点
bert = BertModel.from_pretrained('./bert-base-Chinese')
# 参数都得改
mymodelnew = gru_model.CTTS(input_dim=87).to(device)

mymodelnew.load_state_dict(torch.load('model_state_dict_CTTS_100.pth'))

corporate_list_0 = ['万业企业']

corporate_list = ['万业企业', '万方发展', '万科A', '三湘股份', '上实发展', '上海临港',
                  '上海建工', '世纪星源', '世联地产', '世联行', '世茂集团', '世荣兆业',
                  '东湖高新', '中交地产', '中体产业', '中华企业', '中南建设', '中国中铁',
                  '中国国贸', '中国宝安', '中国恒大', '中国武夷', '中国铁建', '中炬高新',
                  '丰华股份', '云南城投', '五矿地产', '京投发展', '京粮控股', '京能置业',
                  '亿城股份', '佳兆业', '保利发展', '保利地产', '信达地产', '光华控股',
                  '光明地产', '冠城大通', '创兴置业', '北京北辰实业', '北京城建', '华丽家族',
                  '华发股份', '华夏幸福', '华联控股', '华远地产', '南京中北', '南京高科', '南国置业',
                  '南山控股', '卧龙地产', '合肥城建', '嘉宝集团', '城投控股', '外高桥',
                  '大名城', '大悦城', '大港股份', '大龙地产', '天伦置业', '天保基建', '天健集团',
                  '天地源', '天房发展', '天津松江', '宋都股份', '宝龙地产', '实达集团', '市北高新',
                  '广宇发展', '广宇集团', '广汇物流', '建发股份', '张江高科', '新湖中宝', '新黄浦',
                  '栖霞建设', '格力地产', '沙河股份', '泛海建设', '津滨发展', '浦东金桥',
                  '海德股份', '海泰发展', '海航基础', '深房集团', '深振业A', '深深房A',
                  '深物业A', '深长城', '渝开发', '湖南投资', '滨江集团', '珠光控股',
                  '珠江实业', '碧桂园', '福星股份', '空港股份', '粤宏远A', '粤泰股份',
                  '红星发展', '绵世股份', '美好集团', '联发股份', '苏宁环球', '苏州高新',
                  '荣丰控股', '荣安地产', '荣盛发展', '莱茵体育', '莱茵置业', '蓝光发展',
                  '融创中国', '西藏城投', '财信发展', '越秀地产', '迪马股份', '道博股份',
                  '金丰投资', '金地集团', '金科股份', '金融街', '金隅集团', '银亿股份',
                  '阳光股份', '陆家嘴', '陕建股份', '雅居乐集团', '雅戈尔', '顺发恒业',
                  '首开股份', '香江控股', '高新发展', '鲁商置业']
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))


def mean_squared_error(y_true, y_pred):
    return F.mse_loss(y_pred, y_true)

def accuracy(y_true_class, y_pred_class):
    correct = (y_true_class == y_pred_class).sum().item()
    total = y_true_class.size(0)
    return correct, total

def intraclass_correlation(y_true, y_pred):
    # 计算样本均值
    mean_true = torch.mean(y_true, dim=0)
    mean_pred = torch.mean(y_pred, dim=0)

    # 计算内类方差
    var_true = torch.mean((y_true - mean_true) ** 2)
    var_pred = torch.mean((y_pred - mean_pred) ** 2)

    # 计算协方差
    covar = torch.mean((y_true - mean_true) * (y_pred - mean_pred))

    # 计算内类相关系数
    icc = covar / torch.sqrt(var_true * var_pred)

    return icc


def concordance_correlation_coefficient(y_true, y_pred):
    # 计算样本均值
    mean_true = torch.mean(y_true, dim=0)
    mean_pred = torch.mean(y_pred, dim=0)

    # 计算偏差
    true_diff = y_true - mean_true
    pred_diff = y_pred - mean_pred

    # 计算协方差
    covar = torch.mean(true_diff * pred_diff)

    # 计算方差
    var_true = torch.mean(true_diff ** 2)
    var_pred = torch.mean(pred_diff ** 2)

    # 计算一致性相关系数
    ccc = 2 * covar / (var_true + var_pred + (mean_true - mean_pred) ** 2)

    return ccc

mydataset = gru_data.testDataset()
# print(len(mydataset))

dataloader = DataLoader(dataset=mydataset, batch_size=512)
mymodelnew.eval()

avg_mse = 0
ic_mse = 0
ccc_mse = 0
acc_mse = 0
total = 0
correct = 0
with torch.no_grad():
    for data, label, news, policy, wobert, wognn in dataloader:
        # print('这组开始')
        # with torch.no_grad():
        # print(label.shape)
        data = data.to(device)
        news = news.to(device)
        policy = policy.to(device)
        wobert = wobert.to(device)
        wognn = wognn.to(device)
        # graph_data = graph_data.to(device)
        # new_data = new_data.to(device)
        # mask_data = mask_data.to(device)
        # corporate_data = corporate_data.to(device)
        # corporate_mask_data = corporate_mask_data.to(device)
        # policies_data = policies_data.to(device)
        # policy_masks_data = policy_masks_data.to(device)
        label = label.to(device)
        label = label[:, -1, :]
        # print(label)
        # optimizer.zero_grad()
        # output = mymodel(data, news)
        output_100 = mymodelnew(data, news, policy)

        # condition = output_100 >= 0.5
        # # 对满足条件的数进行操作
        # output_100[condition] = (output_100[condition] - 0.5) * 4 + 0.5
        # # 对不满足条件的数进行操作
        # output_100[~condition] = 0.5 - (0.5 - output_100[~condition]) * 4
        # condition = output_200 >= 0.5
        # # 对满足条件的数进行操作
        # output_200[condition] = (output_200[condition] - 0.5) * 4 + 0.5
        # # 对不满足条件的数进行操作
        # output_200[~condition] = 0.5 - (0.5 - output_200[~condition]) * 4
        #
        # condition = output_300 >= 0.5
        # # 对满足条件的数进行操作
        # output_300[condition] = (output_300[condition] - 0.5) * 4 + 0.5
        # # 对不满足条件的数进行操作
        # output_300[~condition] = 0.5 - (0.5 - output_300[~condition]) * 4
        #
        # condition = output_400 >= 0.5
        # # 对满足条件的数进行操作
        # output_400[condition] = (output_400[condition] - 0.5) * 4 + 0.5
        # # 对不满足条件的数进行操作
        # output_400[~condition] = 0.5 - (0.5 - output_400[~condition]) * 4
        #
        # condition = output_500 >= 0.5
        # # 对满足条件的数进行操作
        # output_500[condition] = (output_500[condition] - 0.5) * 4 + 0.5
        # # 对不满足条件的数进行操作
        # output_500[~condition] = 0.5 - (0.5 - output_500[~condition]) * 4

        # print(output.shape)
        # print(output.shape)
        # print(output.shape)
        output_100 = output_100.to(device)
        # print(output.shape)
        # print(output)
        # print(output.shape)
        threshold = 0.5
        y_true_class = (label >= threshold).float()
        y_pred_class = (output_100 >= threshold).float()

        mse = mean_squared_error(output_100, label)
        # print("Mean Squared Error (MSE):", mse.item())

        correct_num, total_num = accuracy(y_true_class, y_pred_class)
        # 计算IC
        icc = intraclass_correlation(output_100, label)
        # print("Intraclass Correlation (IC):", icc.item())

        # 计算CCC
        ccc = concordance_correlation_coefficient(output_100, label)
        # print("Concordance Correlation Coefficient (CCC):", ccc.item())

        avg_mse = avg_mse + mse
        ic_mse = ic_mse + icc
        ccc_mse = ccc_mse + ccc
        correct = correct_num + correct
        total = total_num + total

    # print('有一组')
print("MSE 100: %.4f" % (avg_mse))
print("ICC 100: %.4f" % (ic_mse))
print("CCC 100: %.4f" % (ccc_mse))
print("ACC 100: %.4f" % (correct / total))
