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

device = 'cuda'
# 添加不同类型的节点
bert = BertModel.from_pretrained('./bert-base-Chinese')
# 参数都得改
mymodelnew_100 = gru_model.testmodelwithoutbert(67).to(device)
mymodelnew_200 = gru_model.testmodelwithoutbert(67).to(device)
mymodelnew_300 = gru_model.testmodelwithoutbert(67).to(device)
mymodelnew_400 = gru_model.testmodelwithoutbert(67).to(device)
mymodelnew_500 = gru_model.testmodelwithoutbert(67).to(device)

mymodelnew_100.load_state_dict(torch.load('model_state_dict_total_gru_G_100.pth'))
mymodelnew_100.load_state_dict(torch.load('model_state_dict_total_gru_G_200.pth'))
mymodelnew_100.load_state_dict(torch.load('model_state_dict_total_gru_G_300.pth'))
mymodelnew_100.load_state_dict(torch.load('model_state_dict_total_gru_G_400.pth'))
mymodelnew_100.load_state_dict(torch.load('model_state_dict_total_gru_G_500.pth'))

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
lossfuction = torch.nn.MSELoss()

mydataset = gru_data.testDataset()
# print(len(mydataset))

dataloader = DataLoader(dataset=mydataset, batch_size=512)
mymodelnew_100.eval()
mymodelnew_200.eval()
mymodelnew_300.eval()
mymodelnew_400.eval()
mymodelnew_500.eval()
print_avg_loss_100 = 0
print_avg_loss_200 = 0
print_avg_loss_300 = 0
print_avg_loss_400 = 0
print_avg_loss_500 = 0
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
    output_100 = mymodelnew_100(data, news, policy)
    output_200 = mymodelnew_200(data, news, policy)
    output_300 = mymodelnew_300(data, news, policy)
    output_400 = mymodelnew_400(data, news, policy)
    output_500 = mymodelnew_500(data, news, policy)
    # print(output.shape)
    # print(output.shape)
    # print(output.shape)
    condition = output_100 >= 0.5
    # 对满足条件的数进行操作
    output_100[condition] = (output_100[condition] - 0.5) * 2 + 0.5
    # 对不满足条件的数进行操作
    output_100[~condition] = 0.5 - (0.5 - output_100[~condition]) * 2
    condition = output_200 >= 0.5
    # 对满足条件的数进行操作
    output_200[condition] = (output_200[condition] - 0.5) * 2 + 0.5
    # 对不满足条件的数进行操作
    output_200[~condition] = 0.5 - (0.5 - output_200[~condition]) * 2

    condition = output_300 >= 0.5
    # 对满足条件的数进行操作
    output_300[condition] = (output_300[condition] - 0.5) * 2 + 0.5
    # 对不满足条件的数进行操作
    output_300[~condition] = 0.5 - (0.5 - output_300[~condition]) * 2

    condition = output_400 >= 0.5
    # 对满足条件的数进行操作
    output_400[condition] = (output_400[condition] - 0.5) * 2 + 0.5
    # 对不满足条件的数进行操作
    output_400[~condition] = 0.5 - (0.5 - output_400[~condition]) * 2

    condition = output_500 >= 0.5
    # 对满足条件的数进行操作
    output_500[condition] = (output_500[condition] - 0.5) * 2 + 0.5
    # 对不满足条件的数进行操作
    output_500[~condition] = 0.5 - (0.5 - output_500[~condition]) * 2
    #

    output_100 = output_100.to(device)
    output_200 = output_200.to(device)
    output_300 = output_300.to(device)
    output_400 = output_400.to(device)
    output_500 = output_500.to(device)
    # print(output.shape)
    # print(output)
    # print(output.shape)
    loss_100 = lossfuction(output_100, label)
    loss_200 = lossfuction(output_200, label)
    loss_300 = lossfuction(output_300, label)
    loss_400 = lossfuction(output_400, label)
    loss_500 = lossfuction(output_500, label)
    # print(loss)
    # loss.requires_grad_(True)
    print_avg_loss_100 = print_avg_loss_100 + loss_100
    print_avg_loss_200 = print_avg_loss_200 + loss_200
    print_avg_loss_300 = print_avg_loss_300 + loss_300
    print_avg_loss_400 = print_avg_loss_400 + loss_400
    print_avg_loss_500 = print_avg_loss_500 + loss_500

    # print('有一组')
print("MSE 100: %.4f" % (print_avg_loss_100))
print("MSE 200: %.4f" % (print_avg_loss_200))
print("MSE 300: %.4f" % (print_avg_loss_300))
print("MSE 400: %.4f" % (print_avg_loss_400))
print("MSE 500: %.4f" % (print_avg_loss_500))
