import dgl
import torch
import torch.nn as nn
import dgl.nn.functional as F
import dgl.nn as dglnn
import graph_data
import graph_model
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import attention_data
import attention_model

# 定义模型参数
device = 'cuda'

# 初始化模型
model = attention_model.AttentionGraph().to(device)

# 定义损失函数
loss_func = nn.MSELoss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

mydataset = attention_data.trainDataset()
mydataloader = DataLoader(mydataset, batch_size=32)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for stock, financial, macro, news, policy, label in mydataloader:
        stock = stock.to(device)
        financial = financial.to(device)
        macro = macro.to(device)
        news = news.to(device)
        policy = policy.to(device)
        print(stock.shape)
        output, logits = model(stock, financial, macro, news, policy)
        label = label.to(device)
        # print(output.shape)
        loss = loss_func(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
torch.save(model.state_dict(), 'model_' + 'attention' + '.pth')

# 碧桂园
# 福星股份
# 空港股份
# 粤宏远A
# 粤泰股份
# 红星发展
# 绵世股份
# 美好集团
# 联发股份
# 苏宁环球
# 苏州高新
# 荣丰控股
# 荣安地产
# 荣盛发展
# 莱茵体育
# 莱茵置业
# 蓝光发展
# 融创中国
# 西藏城投
# 财信发展
# 越秀地产
# 迪马股份
# 道博股份
# 金丰投资
# 金地集团
# 金科股份
# 金融街
# 金隅集团
# 银亿股份
# 阳光股份
# 陆家嘴
# 陕建股份
# 雅居乐集团
# 雅戈尔
# 顺发恒业
# 首开股份
# 香江控股
# 高新发展
# 鲁商置业

