import dgl
import torch
import torch.nn as nn
import dgl.nn.functional as F
import dgl.nn as dglnn
import graph_data
import graph_model
import pandas as pd
import numpy as np

print(torch.cuda.is_available())

def split_into_batches(data_list, batch_size):
    cnt = 0
    graph_0 = []
    label_0 = []
    graph_tmp = []
    label_tmp = []
    for graph, label in data_list:
        graph_tmp.append(graph)
        label_tmp.append(label)
        cnt = cnt + 1
        if cnt == batch_size:
            graph_0.append(graph_tmp)
            label_tmp = pd.DataFrame(label_tmp, dtype=np.float32)
            label_tmp = torch.tensor(label_tmp.values).view(1, -1)
            # print(label_tmp.shape)
            label_0.append(label_tmp)
            cnt = 0
            graph_tmp = []
            label_tmp = []

    return zip(graph_0, label_0)

# 定义模型参数
num_classes = 1
hidden_size = 10
device = 'cpu'

# 初始化模型
model = graph_model.HeteroNodeClassifier(hidden_size, num_classes).float().to(device)

# 定义损失函数
loss_func = nn.MSELoss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

mydataset = graph_data.trainDataset()
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for graph, dict, label in mydataset:
        _, logits = model(graph, dict)
        label = torch.tensor(label).view(1, -1).float()
        loss = loss_func(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
torch.save(model.state_dict(), 'model_' + 'graph' + '.pth')

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

