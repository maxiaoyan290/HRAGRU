import dataseter
import model
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from dataseter import y_scaler
import numpy as np
import pandas as pd
from transformers import BertModel,BertTokenizer
from torch.utils.data import DataLoader

mydataset = dataseter.testDataset()

# 添加不同类型的节点
num_nodes = {'Stock': 5, 'Financial': 10, 'Macro': 15, 'News': 20, 'Policy' : 10}  # 节点数量

device = 'cuda'
# 参数都得改
# mymodel = model.testmodel(57).to(device)
transmodel = model.StockTransformer(57, 1).to(device)
# mymodel.load_state_dict(torch.load('model_state_dict_gru.pth'))
# Define LSTM model
lstm_model = model.StockLSTM(57, 64, 1, 2).to(device)
# Define GRU model
gru_model = model.StockGRU(57, 64, 1, 2).to(device)
lstm_model.load_state_dict(torch.load('model_state_dict_lstm.pth'))

outputs = []
labels = []
# mymodel.eval()
lstm_model.eval()
for idx in range(len(mydataset)):
    data, label, news = mydataset[idx]
    # print(label.shape)
    data = data.to(device)
    label = label.to(device)
    news = news.to(device)
    data = data.unsqueeze(0)
    news = news.unsqueeze(0)
    # print(data.shape)
    output = lstm_model(data)
    output = output.item()
    label = label.item()
    outputs.append(output)
    labels.append(label)

x_values = range(1, len(mydataset) + 1)


outputs = np.array(outputs).reshape(1, -1)
labels = np.array(labels).reshape(1, -1)
# condition = outputs >= 0.5
# # 对满足条件的数进行操作
# outputs[condition] = (outputs[condition] - 0.5) * 2 + 0.5
# # 对不满足条件的数进行操作
# outputs[~condition] = 0.5 - (0.5 - outputs[~condition]) * 2

# condition = labels >= 0.5
# # 对满足条件的数进行操作
# labels[condition] = (labels[condition] - 0.5) * 2 + 0.5
# # 对不满足条件的数进行操作
# labels[~condition] = 0.5 - (0.5 - labels[~condition]) * 2

outputs = outputs.flatten().tolist()
labels = labels.flatten().tolist()


# outputs = np.array(outputs).reshape(1, -1)
# outputs = y_scaler.inverse_transform(outputs)
# outputs = outputs.flatten().tolist()
# labels = np.array(labels).reshape(1, -1)
# labels = y_scaler.inverse_transform(labels)
# labels = labels.flatten().tolist()
plt.figure(figsize=(12, 8))
plt.plot(x_values, labels, color='black', label='Label Value')  # 第一个y值用红色表示
plt.plot(x_values, outputs, color='blue', label=' Outputs Value')  # 第二个y值用绿色表示


# 添加图例
plt.legend()

# 添加标题和标签
plt.title('Y Values Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Y Values')

# 保存为图片
plt.savefig('1.png')