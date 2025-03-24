import gru_data
import gru_model
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

mydataset = gru_data.trainDataset()
testdataset = gru_data.testDataset()
# print(len(mydataset))

dataloader = DataLoader(dataset=mydataset, batch_size=512)
testloader = DataLoader(dataset=testdataset, batch_size=512)

device = 'cuda'
# # 参数都得改
# mymodel = model.testmodel(57).to(device)
mymodelnew = gru_model.trainmodel(107).to(device)
# transmodel = model.StockTransformer(57, 1).to(device)
# # Define LSTM model
# lstm_model = model.StockLSTM(57, 64, 1, 2).to(device)
# # Define GRU model
# gru_model = model.StockGRU(57, 64, 1, 2).to(device)
# mymodel.load_state_dict(torch.load('model_state_dict_8.pth'))
lossfuction = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.000164)
optimizer = torch.optim.Adam(mymodelnew.parameters(), lr=0.000164)

epoch = 50
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
        print(output.shape)
        # print(output)
        loss = lossfuction(output, label)
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
        # loss.requires_grad_(True)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        print_val_loss = print_val_loss + loss
    print("Epoch: %d, Loss: %.4f" % (i, print_val_loss))
            # print('有一组')
# torch.save(mymodel.state_dict(), 'model_state_dict_' + 'gru' + '.pth')
    if i == 100:
        torch.save(mymodelnew.state_dict(), 'model_state_dict_' + 'train_gru_100' + '.pth')
    if i == 200:
        torch.save(mymodelnew.state_dict(), 'model_state_dict_' + 'train_gru_200' + '.pth')
    if i == 300:
        torch.save(mymodelnew.state_dict(), 'model_state_dict_' + 'train_gru_300' + '.pth')
    if i == 400:
        torch.save(mymodelnew.state_dict(), 'model_state_dict_' + 'train_gru_400' + '.pth')
    if i == 500:
        torch.save(mymodelnew.state_dict(), 'model_state_dict_' + 'train_gru_500' + '.pth')
    if i == 600:
        torch.save(mymodelnew.state_dict(), 'model_state_dict_' + 'train_gru_600' + '.pth')
    if i == 700:
        torch.save(mymodelnew.state_dict(), 'model_state_dict_' + 'train_gru_700' + '.pth')
    if i == 800:
        torch.save(mymodelnew.state_dict(), 'model_state_dict_' + 'train_gru_800' + '.pth')
    if i == 900:
        torch.save(mymodelnew.state_dict(), 'model_state_dict_' + 'train_gru_900' + '.pth')
    if i == 1000:
        torch.save(mymodelnew.state_dict(), 'model_state_dict_' + 'train_gru_1000' + '.pth')