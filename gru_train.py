import gru_data
import gru_model
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from transformers import BertModel,BertTokenizer
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

corporate_list_0 = ['亿城股份']

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

# f = open('output.txt','w')
# with open ('output.txt','a') as f:
#     f.write('\n')     #换行
#     f.write('111')
#     f.close()

model_name = './bert-base-Chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)

mydataset = gru_data.trainDataset()
testdataset = gru_data.testDataset()

# print(len(mydataset))

dataloader = DataLoader(dataset=mydataset, batch_size=256)
testloader = DataLoader(dataset=testdataset, batch_size=256)

device = 'cuda:1'
# # 参数都得改
# mymodel = model.testmodel(57).to(device)
mymodelnew = gru_model.trainmodel(97).to(device)
# transmodel = model.StockTransformer(57, 1).to(device)
# # Define LSTM model
# lstm_model = model.StockLSTM(57, 64, 1, 2).to(device)
# # Define GRU model
# gru_model = model.StockGRU(57, 64, 1, 2).to(device)
# mymodel.load_state_dict(torch.load('model_state_dict_8.pth'))
lossfuction = torch.nn.MSELoss()

x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
# optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.000164)
optimizer = torch.optim.AdamW(mymodelnew.parameters(), lr=0.000164)

epoch = 5000
mymodelnew.train()
for i in range(epoch):
    print_avg_loss = 0
    print_val_loss = 0
    mymodelnew.train()
    for data, label, news, policy, wobert, wognn in dataloader:
        # print('这组开始')
        # with torch.no_grad():
        #print(label.shape)
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
        output = mymodelnew(data, news, policy)
            #print(output.shape)
            # print(output.shape)
            # print(output.shape)
        output = output.to(device)
        # print(output.shape)
        # print(output)
        loss = lossfuction(output, label)
        torch.nn.utils.clip_grad_norm_(mymodelnew.parameters(), max_norm=1.0)
            # print(loss)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print_avg_loss = print_avg_loss + loss
            # print('有一组')
    print("Epoch: %d, Loss: %.4f" % (i, print_avg_loss))
    for data, label, news, policy, wobert, wognn in testloader:
        # print('这组开始')
        # with torch.no_grad():
        #print(label.shape)
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
        output = mymodelnew(data, news, policy)
            #print(output.shape)
            # print(output.shape)
            # print(output.shape)
        output = output.to(device)
        # print(output.shape)
        # print(output)
        loss = lossfuction(output, label)
            # print(loss)
        loss.requires_grad_(True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mymodelnew.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        print_val_loss = print_val_loss + loss
    print("Epoch: %d, Loss: %.4f" % (i, print_val_loss))
#    if i >= 250:
    torch.save(mymodelnew.state_dict(), 'train_pth_new/model_state_dict_' + 'hragru_' + str(i) + '.pth')

    

#    for corporate in corporate_list:
#        mydataset = gru_data.valDataset(corporate=corporate)
#
#        outputs_gru = []
#        labels = []
#        mymodelnew.eval()
#        for idx in range(len(mydataset)):
#            data, label, news, policy, wobert, wognn = mydataset[idx]
#            data = data.to(device)
#            label = label.to(device)
#            news = news.to(device)
#            policy = policy.to(device)
#            wognn = wognn.to(device)
#            data = data.unsqueeze(0)
#            news = news.unsqueeze(0)
#            policy = policy.unsqueeze(0)
#            wognn = wognn.unsqueeze(0)
#
#            output_gru = mymodelnew(data, news, policy)
#
#            outputs_gru.append(output_gru.item())
#            labels.append(label.item())
#
#        x_values = range(1, len(mydataset) + 1)
#
#        outputs = np.array(outputs_gru).reshape(1, -1)
#        labels = np.array(labels).reshape(1, -1)
#        condition = outputs >= 0.5
#        outputs[condition] = (outputs[condition] - 0.5) * 2 + 0.5
#        outputs[~condition] = 0.5 - (0.5 - outputs[~condition]) * 2
#
#        condition = labels >= 0.5
#        labels[condition] = (labels[condition] - 0.5) * 2 + 0.5
#        labels[~condition] = 0.5 - (0.5 - labels[~condition]) * 2
#
#        labels = labels.flatten().tolist()
#        outputs = outputs.flatten().tolist()
#
#        output_dir = f'corporate_predictions_new/hragru/epoch_{i}'
#        if not os.path.exists(output_dir):  # 检查目录是否存在
#            os.makedirs(output_dir)  # 如果不存在，则创建目录
#
#        file_path = os.path.join(output_dir, f'{corporate}.txt')  # 拼接完整文件路径
#        with open(file_path, 'w') as f:  # 保存文件
#            f.write('Predictions\tLabels\n')
#            for pred, lbl in zip(outputs, labels):
#                f.write(f'{pred}\t{lbl}\n')

# torch.save(mymodel.state_dict(), 'model_state_dict_' + 'gru' + '.pth')
#     torch.save(mymodelnew.state_dict(), 'train_pth/model_state_dict_' + 'train_gru_xlstm_' + str(i) + '.pth')
