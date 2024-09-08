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

mymodelwobert_100 = gru_model.testmodelwithoutbert(67).to(device)
mymodelwobert_200 = gru_model.testmodelwithoutbert(67).to(device)
mymodelwobert_300 = gru_model.testmodelwithoutbert(67).to(device)
mymodelgnn_100 = gru_model.testmodelwithoutgnn(97).to(device)
mymodelgnn_200 = gru_model.testmodelwithoutgnn(97).to(device)
mymodelgnn_300 = gru_model.testmodelwithoutgnn(97).to(device)
mymodelwatt_100 = gru_model.testmodelwithattention(107).to(device)
mymodelwatt_200 = gru_model.testmodelwithattention(107).to(device)
mymodelwatt_300 = gru_model.testmodelwithattention(107).to(device)

mymodelwobert_100.load_state_dict(torch.load('model_state_dict_wobert_gru_100.pth'))
mymodelwobert_200.load_state_dict(torch.load('model_state_dict_wobert_gru_200.pth'))
mymodelwobert_300.load_state_dict(torch.load('model_state_dict_wobert_gru_300.pth'))
mymodelgnn_100.load_state_dict(torch.load('model_state_dict_wognn_gru_100.pth'))
mymodelgnn_200.load_state_dict(torch.load('model_state_dict_wognn_gru_200.pth'))
mymodelgnn_300.load_state_dict(torch.load('model_state_dict_wognn_gru_300.pth'))
mymodelwatt_100.load_state_dict(torch.load('model_state_dict_watt_gru_100.pth'))
mymodelwatt_200.load_state_dict(torch.load('model_state_dict_watt_gru_200.pth'))
mymodelwatt_300.load_state_dict(torch.load('model_state_dict_watt_gru_300.pth'))

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
mymodelwobert_100.eval()
mymodelwobert_200.eval()
mymodelwobert_300.eval()
mymodelgnn_100.eval()
mymodelgnn_200.eval()
mymodelgnn_300.eval()
mymodelwatt_100.eval()
mymodelwatt_200.eval()
mymodelwatt_300.eval()
print_avg_loss_wobert_100 = 0
print_avg_loss_wobert_200 = 0
print_avg_loss_wobert_300 = 0
print_avg_loss_wognn_100 = 0
print_avg_loss_wognn_200 = 0
print_avg_loss_wognn_300 = 0
print_avg_loss_watt_100 = 0
print_avg_loss_watt_200 = 0
print_avg_loss_watt_300 = 0
for data, label, news, policy, wobert, wognn in dataloader:
    # print('这组开始')
    # with torch.no_grad():
    # print(label.shape)
    data = data.to(device)
    news = news.to(device)
    wobert = wobert.to(device)
    wognn = wognn.to(device)
    policy = policy.to(device)
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
    output_wobert_100 = mymodelwobert_100(wobert)
    output_wobert_200 = mymodelwobert_200(wobert)
    output_wobert_300 = mymodelwobert_300(wobert)
    output_wognn_100 = mymodelgnn_100(wognn, news, policy)
    output_wognn_200 = mymodelgnn_200(wognn, news, policy)
    output_wognn_300 = mymodelgnn_300(wognn, news, policy)
    output_watt_100 = mymodelwatt_100(data, news, policy)
    output_watt_200 = mymodelwatt_200(data, news, policy)
    output_watt_300 = mymodelwatt_300(data, news, policy)
    output_wobert_100 = output_wobert_100.to(device)
    output_wobert_200 = output_wobert_200.to(device)
    output_wobert_300 = output_wobert_300.to(device)
    output_wognn_100 = output_wognn_100.to(device)
    output_wognn_200 = output_wognn_200.to(device)
    output_wognn_300 = output_wognn_300.to(device)
    output_watt_100 = output_watt_100.to(device)
    output_watt_200 = output_watt_200.to(device)
    output_watt_300 = output_watt_300.to(device)
    loss_wobert_100 = lossfuction(output_wobert_100, label)
    loss_wobert_200 = lossfuction(output_wobert_200, label)
    loss_wobert_300 = lossfuction(output_wobert_300, label)
    loss_wognn_100 = lossfuction(output_wognn_100, label)
    loss_wognn_200 = lossfuction(output_wognn_200, label)
    loss_wognn_300 = lossfuction(output_wognn_300, label)
    loss_watt_100 = lossfuction(output_watt_100, label)
    loss_watt_200 = lossfuction(output_watt_200, label)
    loss_watt_300 = lossfuction(output_watt_300, label)
    # print(loss)
    print_avg_loss_wobert_100 = print_avg_loss_wobert_100 + loss_wobert_100
    print_avg_loss_wobert_200 = print_avg_loss_wobert_200 + loss_wobert_200
    print_avg_loss_wobert_300 = print_avg_loss_wobert_300 + loss_wobert_300
    print_avg_loss_wognn_100 = print_avg_loss_wognn_100 + loss_wognn_100
    print_avg_loss_wognn_200 = print_avg_loss_wognn_200 + loss_wognn_200
    print_avg_loss_wognn_300 = print_avg_loss_wognn_300 + loss_wognn_300
    print_avg_loss_watt_100 = print_avg_loss_watt_100 + loss_watt_100
    print_avg_loss_watt_200 = print_avg_loss_watt_200 + loss_watt_200
    print_avg_loss_watt_300 = print_avg_loss_watt_300 + loss_watt_300

    # print('有一组')
print("MSE wobert_100: %.4f" % (print_avg_loss_wobert_100))
print("MSE wobert_200: %.4f" % (print_avg_loss_wobert_200))
print("MSE wobert_300: %.4f" % (print_avg_loss_wobert_300))
print("MSE wognn_100: %.4f" % (print_avg_loss_wognn_100))
print("MSE wognn_200: %.4f" % (print_avg_loss_wognn_200))
print("MSE wognn_300: %.4f" % (print_avg_loss_wognn_300))
print("MSE watt_100: %.4f" % (print_avg_loss_watt_100))
print("MSE watt_200: %.4f" % (print_avg_loss_watt_200))
print("MSE watt_300: %.4f" % (print_avg_loss_watt_300))