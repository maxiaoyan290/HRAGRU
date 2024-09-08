import bert_data
import bert_model
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertModel,BertTokenizer
from torch_geometric.data import DataLoader

model_name = './bert-base-Chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)

mydataset = bert_data.trainDataset(tokenizer)

mydataloader = DataLoader(mydataset, batch_size=256)
device = 'cuda:1'
# 添加不同类型的节点
bert = BertModel.from_pretrained('./bert-base-Chinese')
# 参数都得改
mymodel = bert_model.BertGNNGru(bert_with_layer_size=10, news_size=10, bert=bert, num_heads=1).to(device)
lossfuction = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(mymodel.parameters(),lr=2e-5)
epoch = 2
for i in range(epoch):
    print_avg_loss = 0
    for new_data, mask_data, corporate_data, corporate_mask_data, label in mydataloader:
        #print(label.shape)
        optimizer.zero_grad()
        new_data = new_data.to(device)
        mask_data = mask_data.to(device)
        corporate_data = corporate_data.to(device)
        corporate_mask_data = corporate_mask_data.to(device)
        label = label.to(device)

        output = mymodel(new_data, mask_data, corporate_data, corporate_mask_data)
        #print(output.shape)
        loss = lossfuction(output, label)
        loss.backward()
        optimizer.step()
        print_avg_loss = print_avg_loss + loss
        # print('一组了')
    print("Epoch: %d, Loss: %.4f" % (i, print_avg_loss))
    print_avg_loss = 0
torch.save(mymodel.state_dict(), 'model_' + 'bert' + '.pth')