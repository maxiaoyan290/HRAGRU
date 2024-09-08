import dataseter
import model
import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from transformers import BertModel,BertTokenizer
from torch.utils.data import DataLoader

# f = open('output.txt','w')
# with open ('output.txt','a') as f:
#     f.write('\n')     #换行
#     f.write('111')
#     f.close()

model_name = './bert-base-Chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)

mydataset = dataseter.trainDataset()
# print(len(mydataset))

dataloader = DataLoader(dataset=mydataset, batch_size=512)

# 添加不同类型的节点
num_nodes = {'Stock': 5, 'Financial': 10, 'Macro': 15, 'News': 20, 'Policy' : 10}  # 节点数量
# 假设每个节点类型对之间都存在边
edge_index_list = []
for src_type in num_nodes.keys():
    for dst_type in num_nodes.keys():
        if src_type != dst_type:
            # 添加边到异构图中
            edge_index_list.append((f'{src_type}_x', f'{src_type}_x_{dst_type}_x', f'{dst_type}_x'))
# # 创建metadata
metadata = (['Stock_x', 'Financial_x', 'Macro_x', 'News_x', 'Policy_x'], edge_index_list)
input_dict = {'Stock_x': 10, 'Financial_x': 16, 'Macro_x': 21, 'News_x': 20, 'Policy_x' : 10}
device = 'cuda'
bert = BertModel.from_pretrained('./bert-base-Chinese')
# 参数都得改
mymodel = model.testmodel(57).to(device)
mymodelnew = model.testmodelnew(57).to(device)
transmodel = model.StockTransformer(57, 1).to(device)
# Define LSTM model
lstm_model = model.StockLSTM(57, 64, 1, 2).to(device)
# Define GRU model
gru_model = model.StockGRU(57, 64, 1, 2).to(device)
# mymodel.load_state_dict(torch.load('model_state_dict_8.pth'))
lossfuction = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.000164)
optimizer = torch.optim.Adam(mymodelnew.parameters(), lr=0.000164)

epoch = 100
mymodelnew.train()
for i in range(epoch):
    print_avg_loss = 0
    mymodelnew.train()
    for data, label, news in dataloader:
        # print('这组开始')
        # with torch.no_grad():
        #print(label.shape)
        data = data.to(device)
        news = news.to(device)
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
        output = mymodelnew(data, news)
            #print(output.shape)
            # print(output.shape)
            # print(output.shape)
        output = output.to(device)
        # print(output.shape)
        #print(output)
            # print(output.shape)
        loss = lossfuction(output, label)
            # print(loss)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print_avg_loss = print_avg_loss + loss
            # print('有一组')
    print("Epoch: %d, Loss: %.4f" % (i, print_avg_loss))
# torch.save(mymodel.state_dict(), 'model_state_dict_' + 'gru' + '.pth')
torch.save(mymodelnew.state_dict(), 'model_state_dict_' + 'gru_new' + '.pth')
