import os

'''
下面代码用于获取所有公司名
'''
# def corporate_list(file_path):
#     mylist = []
#     for file in os.listdir(file_path):
#         mylist.append(file.split('.')[0])
#     print(mylist)

'''
训练测试
'''
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
#
#
# # 构建数据集类
# class MyDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         a, b, c, d, e = self.data[index]
#         a = torch.tensor(a, dtype=torch.float32)
#         b = torch.tensor(b, dtype=torch.float32)
#         c = torch.tensor(c, dtype=torch.float32)
#         d = torch.tensor(d, dtype=torch.float32)
#         e = [torch.tensor(item, dtype=torch.float32) for item in e]
#         return a, b, c, d, e
#
#
# # 构建模型类
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.fc_a = nn.Linear(4, 64)  # 输入4维，输出64维
#         self.fc_b = nn.Linear(4, 64)
#         self.fc_c = nn.Linear(4, 64)
#         self.fc_d = nn.Linear(4, 64)
#         self.fc_e = nn.Linear(4, 64)
#
#     def forward(self, a, b, c, d, e):
#         a_out = torch.relu(self.fc_a(a))
#         b_out = torch.relu(self.fc_b(b))
#         c_out = torch.relu(self.fc_c(c))
#         d_out = torch.relu(self.fc_d(d))
#
#         # 对e中的每个tensor过fc层并求平均
#         e_out = torch.stack([torch.relu(self.fc_e(x)) for x in e])
#         e_avg_out = torch.mean(e_out, dim=0)
#
#         # 返回所有结果的平均值
#         return (a_out + b_out + c_out + d_out + e_avg_out) / 5
#
#
# # 构建数据集
# data = [(torch.rand(4), torch.rand(4), torch.rand(4), torch.rand(4), [torch.rand(4) for _ in range(5)]) for _ in
#         range(100)]
# dataset = MyDataset(data)
#
# # 创建模型并将其移至GPU
# model = MyModel()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# # 构建数据加载器
# batch_size = 16
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# # 示例训练代码
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss()
#
# for epoch in range(5):
#     for batch in dataloader:
#         # 将a、b、c、d移至GPU
#         a, b, c, d = [item.to(device) for item in batch[:4]]
#         # 将e中的每个张量移至GPU
#         e = [item.to(device) for item in batch[4]]
#
#         #batch = [item.to(device) for item in batch]
#         optimizer.zero_grad()
#         output = model(a, b, c, d, e)
#         loss = criterion(output, torch.zeros_like(output))  # 示例损失函数，这里假设目标是0
#         loss.backward()
#         optimizer.step()
#
#     print(f"Epoch [{epoch + 1}/5], Loss: {loss.item():.4f}")

'''
图神经网络测试
'''
'''
no.1
'''

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import HANConv
# from torch_geometric.data import Data
#
#
# def create_hetero_graph():
#     # 创建异构图对象
#     data = Data()
#
#     # 添加不同类型的节点
#     num_nodes = 4  # 四个节点
#     node_sizes = [5, 10, 15, 20]  # 节点特征大小
#     for i in range(num_nodes):
#         node_type = f'node_type_{i}'
#         # 随机初始化节点特征，注意张量大小不一样
#         x = torch.randn(1, node_sizes[i])
#         data[f'{node_type}_x'] = x
#
#     # 添加边到异构图
#     # 在这个示例中，假设每个节点类型对之间有两条边
#     edges = [((0, 1), (1, 0)), ((2, 3), (3, 2))]
#     for edge in edges:
#         src_types, dst_types = edge
#         for src_type, dst_type in zip(src_types, dst_types):
#             src_type = f'node_type_{src_type}'
#             dst_type = f'node_type_{dst_type}'
#             # 添加边到异构图中
#             src_key = f'{src_type}_x'
#             dst_key = f'{dst_type}_x'
#             if 'edge_index' not in data:
#                 data.edge_index = []
#             data.edge_index.append((src_key, dst_key))
#
#     return data
#
#
# # 定义HAN模型
# class HANModel(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_types, metadata):
#         super(HANModel, self).__init__()
#         self.metadata = metadata
#         self.conv1 = HANConv(in_channels, hidden_channels, heads=1, num_types=num_types, metadata=metadata)
#         self.conv2 = HANConv(hidden_channels, out_channels, heads=1, num_types=num_types, metadata=metadata)
#
#     def forward(self, data):
#         x_dict = {key: data[key] for key in data.keys() if key.endswith('_x')}
#         edge_index_dict = {(src_key, dst_key): data.edge_index[idx]
#                            for idx, (src_key, dst_key) in enumerate(data.edge_index)}
#
#         # 第一轮消息传递
#         x_list = []
#         for (src_key, dst_key), edge_index in edge_index_dict.items():
#             x = self.conv1(x_dict[src_key], edge_index,
#                            torch.tensor([int(src_key[-3])]), torch.tensor([int(dst_key[-3])]))
#             x_list.append(x)
#
#         # 将所有节点的特征拼接在一起
#         x = torch.cat(x_list, dim=0)
#         x = F.relu(x)
#
#         # 第二轮消息传递
#         out_dict = {}
#         for (src_key, dst_key), edge_index in edge_index_dict.items():
#             x = self.conv2(x, edge_index,
#                            torch.tensor([int(src_key[-3])]), torch.tensor([int(dst_key[-3])]))
#             out_dict[dst_key] = x
#
#         return out_dict
#
#
# # 创建异构图数据
# graph_data = create_hetero_graph()
#
# # 创建metadata
# metadata = (['node_type_0', 'node_type_1', 'node_type_2', 'node_type_3'],
#             [('node_type_0', 'edge_type_1', 'node_type_1'), ('node_type_2', 'edge_type_2', 'node_type_3')])
# input_dict = {'node_type_1': 5, 'node_type_2': 10, 'node_type_3': 15, 'node_type_4': 20}
#
# # 实例化HAN模型
# model = HANModel(in_channels=input_dict, hidden_channels=64, out_channels=32, num_types=4, metadata=metadata)
#
# # 执行前向传播
# output = model(graph_data)
#
# print("Output dictionary:", output)  # 输出图的表示

'''
no.2
'''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import HANConv
# from torch_geometric.data import Data, DataLoader
#
# edge_index_list = []
# num_nodes = {'author': 5, 'paper': 10, 'field': 15, 'institution': 20, 'mxy' : 100}  # 节点数量
# for src_type in num_nodes.keys():
#     for dst_type in num_nodes.keys():
#         if src_type != dst_type:
#             # 添加边到异构图中
#             edge_index_list.append((f'{src_type}_x', f'{src_type}_x_{dst_type}_x', f'{dst_type}_x'))
#
# def create_hetero_graph():
#     # 创建异构图对象
#     data = Data()
#
#     # 添加不同类型的节点
#     num_nodes = {'author': 5, 'paper': 10, 'field': 15, 'institution': 20}  # 节点数量
#     for node_type, num_node in num_nodes.items():
#         # 添加节点特征，这里随机初始化节点特征，张量大小不一样
#         x = torch.randn(1, num_node)
#         data[f'{node_type}_x'] = x
#
#     # 添加边到异构图
#     # 假设每个节点类型对之间都存在边
#     edge_index_list_ano = []
#     for src_type in num_nodes.keys():
#         for dst_type in num_nodes.keys():
#             if src_type != dst_type:
#                 # 添加边到异构图中
#                 #edge_index_list.append((f'{src_type}_x', f'{src_type}_x_{dst_type}_x', f'{dst_type}_x'))
#                 edge_index_list_ano.append((f'{src_type}_x', f'{dst_type}_x', [[0, 0]]))  # 仅包含一个节点的索引
#
#     # 构建边索引字典
#     edge_index_dict = {}
#     for src_key, dst_key, edge_index in edge_index_list_ano:
#         # if (src_key, dst_key) not in edge_index_dict:
#         #     edge_index_dict[(src_key, dst_key)] = []
#         # edge_index_dict[(src_key, dst_key)].append((src_key, dst_key))
#         edge_index_dict[(src_key, src_key + '_' + dst_key, dst_key)] = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#
#
#     data.edge_index_dict = edge_index_dict
#
#     return data
#
#
# # 定义HAN模型
# class HANModel(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_types, metadata):
#         super(HANModel, self).__init__()
#         self.metadata = metadata
#         self.conv1 = HANConv(in_channels, hidden_channels, heads=1, num_types=num_types, metadata=self.metadata)
#         self.conv2 = HANConv(hidden_channels, out_channels, heads=1, num_types=num_types, metadata=self.metadata)
#
#     def forward(self, data):
#         x_dict = {}
#         # 构建节点特征字典，并包含完整的节点键
#         for key in data.keys():
#             if key.endswith('_x'):
#                 x_dict[key] = data[key]
#         edge_dict = data.edge_index_dict
#
#         print(x_dict)
#         print(edge_dict)
#
#         x = self.conv1(x_dict, edge_dict)
#         out_dict = self.conv2(x, edge_dict)
#         sum_tensor = torch.zeros(1, 10)
#         for tensor in out_dict.values():
#             sum_tensor += tensor
#         num_tensors = len(out_dict)
#         average_tensor = sum_tensor / num_tensors
#
#         return average_tensor
#
#
# # 创建异构图数据
# graph_data = create_hetero_graph()
#
# # 创建metadata
# metadata = (['author_x', 'paper_x', 'field_x', 'institution_x', 'mxy_x'], edge_index_list)
# print(metadata)
# input_dict = {'author_x': 5, 'paper_x': 10, 'field_x': 15, 'institution_x': 20, 'mxy_x': 30}
#
# # 实例化HAN模型
# model = HANModel(in_channels=input_dict, hidden_channels=20, out_channels=10, num_types=5, metadata=metadata)
#
# # 执行前向传播
# output = model(graph_data)
#
# print("Output dictionary:", output)  # 输出图的表示

'''
no。3
'''
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from transformers import BertModel, BertTokenizer
# from tqdm import tqdm
#
#
# # 定义自定义的Dataset类
# class CustomDataset(Dataset):
#     def __init__(self, data, max_length):
#         self.data = data
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         ten_dim_vector, array = self.data[idx]
#         return ten_dim_vector, array
#
#
# # 创建数据集
# data = []
# for _ in range(10):
#     sub_data = []
#     for i in range(20):
#         #ten_dim_vector = torch.randn(10)
#         num_vectors = np.random.randint(1, 6)  # 随机生成向量数量
#         array = [torch.randn(5) for _ in range(num_vectors)]
#         sub_data.append((array))
#     data.append(sub_data)
#
# val = data
# val = np.array(val)
# print(val.shape)
# data = torch.tensor(val)
# print(data.shape)
#
# # 初始化BERT模型和分词器
# model_name = 'bert-base-uncased'  # 或者其他预训练的BERT模型
# tokenizer = BertTokenizer.from_pretrained(model_name)
#
# # 创建自定义Dataset实例
# max_length = 64  # 假设最大长度为64
# dataset = CustomDataset(data, max_length)
#
# # 创建DataLoader
# batch_size = 1
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#
# # 初始化BERT模型
# model = BertModel.from_pretrained(model_name)
#
# # 将BERT模型移动到GPU上
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
#
# # 定义微调模型
# class FineTunedBert(nn.Module):
#     def __init__(self, bert_model, input_size, output_size):
#         super(FineTunedBert, self).__init__()
#         self.bert_model = bert_model
#         self.fc = nn.Linear(input_size, output_size)
#
#     def forward(self, inputs):
#         with torch.no_grad():
#             outputs = self.bert_model(**inputs)
#         pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
#         return self.fc(pooled_output)
#
#
# # 定义微调模型的输入输出大小
# input_size = model.config.hidden_size
# output_size = 1  # 暂时设定为1，您可以根据您的任务调整输出大小
#
# # 创建微调模型实例
# fine_tuned_model = FineTunedBert(model, input_size, output_size).to(device)
#
# # 定义优化器和损失函数
# optimizer = torch.optim.AdamW(fine_tuned_model.parameters(), lr=1e-5)
# criterion = nn.MSELoss()
#
# # 定义训练循环
# epochs = 3  # 假设训练3个epoch
# for epoch in range(epochs):
#     fine_tuned_model.train()
#     running_loss = 0.0
#     for ten_dim_vectors, arrays in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
#         optimizer.zero_grad()
#
#         # 对array中的每个向量过BERT并求平均
#         inputs = []
#         for i in range(arrays.size(1)):
#             tokenized_inputs = tokenizer(arrays[0, i], padding=True, truncation=True, max_length=max_length,
#                                          return_tensors='pt')
#             inputs.append(tokenized_inputs)
#         inputs = {key: torch.cat([inp[key] for inp in inputs]) for key in inputs[0].keys()}
#         inputs = {key: value.to(device) for key, value in inputs.items()}
#         outputs = fine_tuned_model(inputs)
#
#         # 假设标签全部为0
#         labels = torch.zeros_like(outputs)
#         labels = labels.to(device)
#
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch {epoch + 1} Loss: {running_loss / len(dataloader)}")

# 在此之后，您可以将fine_tuned_model用于您的特定任务

'''
模型测试
'''
#
# import dataseter
# import torch
# from torch_geometric.data import DataLoader
# from torch_geometric.data import HeteroData
# import numpy as np
# import pandas as pd
# from transformers import BertModel,BertTokenizer
# from torch_geometric.data import Batch
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import BertModel, BertConfig, BertLayer
# from torch_geometric.nn import GATConv,HANConv
# import math
# from torch.autograd import Variable
# import pandas as pd
# import numpy as np
#
# '''
# 可能实际使用中没有batch_size，后面要加上
# '''
# num_nodes = {'Stock': 5, 'Financial': 10, 'Macro': 15, 'News': 20, 'Policy': 100}  # 节点数量
# edge_index_list_ano = []
# for src_type in num_nodes.keys():
#     for dst_type in num_nodes.keys():
#         if src_type != dst_type:
#             # 添加边到异构图中
#             edge_index_list_ano.append((f'{src_type}_x', f'{dst_type}_x', [[0, 0]]))  # 仅包含一个节点的索引
#
# # 构建边索引字典
# edge_index_dict = {}
# for src_key, dst_key, edge_index in edge_index_list_ano:
#     edge_index_dict[(src_key, src_key + '_' + dst_key, dst_key)] = torch.tensor(edge_index,
#                                                                                 dtype=torch.long).t().contiguous()
#
# # 定义HAN模型
# class HANModel(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_types, metadata):
#         super(HANModel, self).__init__()
#         self.metadata = metadata
#         self.conv1 = HANConv(in_channels, hidden_channels, heads=1, metadata=self.metadata)
#         self.conv2 = HANConv(hidden_channels, out_channels, heads=1, metadata=self.metadata)
#
#     def forward(self, data):
#         x_dict = {}
#         # 构建节点特征字典，并包含完整的节点键
#         for key in data.keys():
#             if '_x_' not in key and key != 'edge_index_dict':
#                 x_dict[key] = data[key]
#         edge_dict = data.edge_index_dict
#
#         # 将 x_dict 中的张量移到 CUDA 上
#         x_dict_cuda = {key: tensor.to(device) for key, tensor in x_dict.items()}
#
#         # 将 edge_dict 中的张量移到 CUDA 上
#         edge_dict_cuda = {key: tensor.to(device) for key, tensor in edge_dict.items()}
#         x = self.conv1(x_dict_cuda, edge_dict_cuda)
#         out_dict = self.conv2(x, edge_dict_cuda)
#         # print(out_dict)
#
#         sum_tensor = torch.zeros(1, 10).cuda()
#         for tensor in out_dict.values():
#             sum_tensor += tensor
#         num_tensors = len(out_dict)
#         print(sum_tensor)
#         average_tensor = sum_tensor / num_tensors
#
#         return average_tensor
#
# def custom_collate(batch):
#     # 提取每个元组中的图数据
#     graph_data = [item[1] for item in batch]
#     #print(graph_data)
#
#     # 扁平化图数据
#     flat_graph_data = []
#     for sublist in graph_data:
#         flat_graph_data.extend(sublist)
#     data = [item[0] for item in batch]
#     data = torch.stack(data)
#     # 提取其他类型的张量数据
#     # other_data = [item[0] for item in batch] + [item[2:] for item in batch]
#     return {'graph' : flat_graph_data, 'data' : data}
#     #return flat_graph_data, flat_graph_data, flat_graph_data, flat_graph_data, flat_graph_data, flat_graph_data, flat_graph_data, flat_graph_data, flat_graph_data
# model_name = './bert-base-Chinese'
# tokenizer = BertTokenizer.from_pretrained(model_name)
#
# # mydataset = dataseter.trainDataset(tokenizer)
# #
# # data_loader = DataLoader(mydataset, batch_size=4, collate_fn=custom_collate)
#
# # 添加不同类型的节点
# num_nodes = {'Stock': 5, 'Financial': 10, 'Macro': 15, 'News': 20, 'Policy' : 10}  # 节点数量
# # 假设每个节点类型对之间都存在边
# edge_index_list = []
# for src_type in num_nodes.keys():
#     for dst_type in num_nodes.keys():
#         if src_type != dst_type:
#             # 添加边到异构图中
#             edge_index_list.append((f'{src_type}_x', f'{src_type}_x_{dst_type}_x', f'{dst_type}_x'))
# # # 创建metadata
# metadata = (['Stock_x', 'Financial_x', 'Macro_x', 'News_x', 'Policy_x'], edge_index_list)
# input_dict = {'Stock_x': 10, 'Financial_x': 16, 'Macro_x': 21, 'News_x': 20, 'Policy_x' : 10}
# device = 'cuda'
# bert = BertModel.from_pretrained('./bert-base-Chinese')
# # 参数都得改
# han = HANModel(in_channels=input_dict, hidden_channels=20, out_channels= 10,metadata=metadata, num_types=5)
# lossfuction = torch.nn.MSELoss()
# epoch = 2
# for i in range(epoch):
#     print_avg_loss = 0
#     data_graph = HeteroData()
#     # 添加不同类型的节点
#
#     num_nodes = {'Stock': 5, 'Financial': 10, 'Macro': 15, 'News': 20, 'Policy': 10}  # 节点数量
#     # for node_type, num_node in num_nodes.items():
#     # 添加节点特征，这里随机初始化节点特征，张量大小不一样
#     # x = torch.randn(1, num_node)
#     data_graph[f'Stock_x'] = torch.randn(1, 10)
#     data_graph[f'Financial_x'] = torch.randn(1, 16)
#     data_graph[f'Macro_x'] = torch.randn(1, 21)
#     data_graph[f'News_x'] = torch.randn(0, 20)
#     data_graph[f'Policy_x'] = torch.randn(0, 10)
#
#     for src_type in num_nodes.keys():
#         for dst_type in num_nodes.keys():
#             if src_type != dst_type:
#                 # 添加边到异构图中
#                 if src_type == 'News' or dst_type == 'News':
#                     data_graph[f'{src_type}_x_{dst_type}_x'] = torch.zeros(0, 10)
#                     continue
#                 elif src_type == 'Policy' or dst_type == 'Policy':
#                     data_graph[f'{src_type}_x_{dst_type}_x'] = torch.zeros(0, 10)
#                     continue
#                 else:
#                     data_graph[f'{src_type}_x_{dst_type}_x'] = torch.zeros(1, 10)
#
#     # 添加边到异构图
#     # 假设每个节点类型对之间都存在边
#     edge_index_list = []
#     edge_index_list_ano = []
#     for src_type in num_nodes.keys():
#         for dst_type in num_nodes.keys():
#             if src_type != dst_type:
#                 # 添加边到异构图中
#                 if src_type == 'News' or dst_type == 'News':
#                     edge_index_list_ano.append((f'{src_type}_x', f'{dst_type}_x', torch.randn(2, 0)))  # 仅包含一个节点的索引
#                     continue
#                 elif src_type == 'Policy' or dst_type == 'Policy':
#                     edge_index_list_ano.append((f'{src_type}_x', f'{dst_type}_x', torch.randn(2, 0)))  # 仅包含一个节点的索引
#                     continue
#                 else:
#                     edge_index_list_ano.append((f'{src_type}_x', f'{dst_type}_x', torch.tensor([[0], [0]], dtype=torch.long)))  # 仅包含一个节点的索引
#
#     # 构建边索引字典
#     edge_index_dict = {}
#     for src_key, dst_key, edge_index in edge_index_list_ano:
#         edge_index_dict[(src_key, src_key + '_' + dst_key, dst_key)] = edge_index
#
#     data_graph.edge_index_dict = edge_index_dict
#
#     data_graph_1 = HeteroData()
#     # 添加不同类型的节点
#
#     num_nodes = {'Stock': 5, 'Financial': 10, 'Macro': 21, 'News': 20, 'Policy': 10}  # 节点数量
#     # for node_type, num_node in num_nodes.items():
#     # 添加节点特征，这里随机初始化节点特征，张量大小不一样
#     # x = torch.randn(1, num_node)
#     data_graph_1[f'Stock_x'] = torch.randn(0, 10)
#     data_graph_1[f'Financial_x'] = torch.randn(0, 16)
#     data_graph_1[f'Macro_x'] = torch.randn(1, 21)
#     data_graph_1[f'News_x'] = torch.randn(1, 20)
#     data_graph_1[f'Policy_x'] = torch.randn(1, 10)
#
#     for src_type in num_nodes.keys():
#         for dst_type in num_nodes.keys():
#             if src_type != dst_type:
#                 if src_type == 'Stock' or dst_type == 'Stock':
#                     data_graph_1[f'{src_type}_x_{dst_type}_x'] = torch.zeros(0, 10)
#                     continue
#                 elif src_type == 'Financial' or dst_type == 'Financial':
#                     data_graph_1[f'{src_type}_x_{dst_type}_x'] = torch.zeros(0, 10)
#                     continue
#                 else:
#                     data_graph_1[f'{src_type}_x_{dst_type}_x'] = torch.zeros(1, 10)
#
#     # 添加边到异构图
#     # 假设每个节点类型对之间都存在边
#     edge_index_list = []
#     edge_index_list_ano = []
#     for src_type in num_nodes.keys():
#         for dst_type in num_nodes.keys():
#             if src_type != dst_type:
#                 if src_type == 'Stock' or dst_type == 'Stock':
#                     edge_index_list_ano.append((f'{src_type}_x', f'{dst_type}_x', torch.randn(2, 0)))  # 仅包含一个节点的索引
#                     continue
#                 elif src_type == 'Financial' or dst_type == 'Financial':
#                     edge_index_list_ano.append((f'{src_type}_x', f'{dst_type}_x', torch.randn(2, 0)))  # 仅包含一个节点的索引
#                     continue
#                 else:
#                     edge_index_list_ano.append((f'{src_type}_x', f'{dst_type}_x', torch.tensor([[0], [0]], dtype=torch.long)))  # 仅包含一个节点的索引
#
#     # 构建边索引字典
#     edge_index_dict = {}
#     for src_key, dst_key, edge_index in edge_index_list_ano:
#         edge_index_dict[(src_key, src_key + '_' + dst_key, dst_key)] = edge_index
#
#     data_graph_1.edge_index_dict = edge_index_dict
#
#     graph_data = [data_graph, data_graph_1, data_graph, data_graph_1]
#     data_loader = DataLoader(graph_data, batch_size=2)
#     for batch in data_loader:
#         # print(graph_data)
#         with torch.no_grad():
#             # data = data.to(device)
#                 #graph_data = [graph.to(device) for graph in graph_data]
#             #print(graph_data)
#             # graph_data = Batch.from_data_list(graph_data)
#             # graph_data.to(device)
#             batch.to(device)
#             han.to(device)
#             output = han(batch)
#             print(output)
#     print("Epoch: %d, Loss: %.4f" % (i, print_avg_loss))
#     print_avg_loss = 0

'''
main 函数
'''
# if __name__ == '__main__':
#     corporate_list('news_data')

# import torch
# from torch_geometric.data import HeteroData, Batch
#
# # 创建第一个图的节点特征和边索引（包含abc类数据节点）
# data_graph_abc = HeteroData()
# num_nodes_abc = {'Stock': 5, 'Financial': 10, 'Macro': 15}  # 节点数量
#
# # 添加不同类型的节点特征
# for node_type, num_node in num_nodes_abc.items():
#     data_graph_abc[f'{node_type}_x'] = torch.randn(1, num_node)
#
# # 添加边到异构图
# edge_index_list_abc = []
# for src_type in num_nodes_abc.keys():
#     for dst_type in num_nodes_abc.keys():
#         if src_type != dst_type:
#             # 添加边到异构图中
#             edge_index_list_abc.append((f'{src_type}_x', f'{src_type}_x_{dst_type}_x', f'{dst_type}_x'))
#
# # 构建边索引字典
# edge_index_dict_abc = {}
# for src_key, key, dst_key in edge_index_list_abc:
#     edge_index_dict_abc[(src_key, key, dst_key)] = torch.tensor([[0], [0]], dtype=torch.long).t().contiguous()
#
# data_graph_abc.edge_index_dict = edge_index_dict_abc
#
# # 创建第二个图的节点特征和边索引（包含cde类数据节点）
# data_graph_cde = HeteroData()
# num_nodes_cde = {'Macro': 21, 'News': 20, 'Policy': 10}  # 节点数量
#
# # 添加不同类型的节点特征
# for node_type, num_node in num_nodes_cde.items():
#     data_graph_cde[f'{node_type}_x'] = torch.randn(1, num_node)
#
# # 添加边到异构图
# edge_index_list_cde = []
# for src_type in num_nodes_cde.keys():
#     for dst_type in num_nodes_cde.keys():
#         if src_type != dst_type:
#             # 添加边到异构图中
#             edge_index_list_cde.append((f'{src_type}_x', f'{src_type}_x_{dst_type}_x', f'{dst_type}_x'))
#
# # 构建边索引字典
# edge_index_dict_cde = {}
# for src_key, key, dst_key in edge_index_list_cde:
#     edge_index_dict_cde[(src_key, key, dst_key)] = torch.tensor([[0], [0]], dtype=torch.long).t().contiguous()
#
# data_graph_cde.edge_index_dict = edge_index_dict_cde
#
# # 将两个数据对象放入一个列表中
# graph_data = [data_graph_abc, data_graph_cde]
# print(graph_data)
# # 使用 Batch.from_data_list() 创建一个 Batch 对象
# batch_data = Batch.from_data_list(graph_data)

# import torch
#
# # 创建一个示例张量
# tensor = torch.tensor([
#     [[0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
#      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
#
#     [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
#
#     [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# ])
# sum_tensor = torch.sum(tensor, dim=1)
# # 在 dim=1 上计算非零元素的个数
# nonzero_counts = torch.count_nonzero(tensor, dim=1)
# nonzero_counts = torch.where(nonzero_counts == 0, torch.tensor(1), nonzero_counts)
#
# print("每个 slice 中非零元素的个数：")
# print(sum_tensor / nonzero_counts)
#
# import numpy as np
# from sklearn.cluster import KMeans
#
#
# def rearrange_Y(X, Y, n_clusters):
#     # 对 X 进行聚类
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(X)
#
#     # 获取聚类标签
#     labels = kmeans.labels_
#     new_X = kmeans.cluster_centers_
#
#     # 初始化新的 Y 矩阵
#     new_Y = np.zeros_like(Y)  # 创建与 Y 相同形状的全零矩阵
#
#     # 对 Y 中的每个样本按照 X 的聚类结果重新排列
#     for i in range(n_clusters):
#         cluster_indices = np.where(labels == i)[0]  # 获取属于第 i 个簇的样本索引
#         new_Y[cluster_indices] = Y[labels == i]  # 将对应聚类标签的 Y 样本重新排列到新的 Y 矩阵中
#
#     return new_Y, new_X
#
#
# # 示例用法
# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 三行分别是123
# Y = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])  # 三行分别是456
# n_clusters = 2
#
# new_Y, new_X = rearrange_Y(X, Y, n_clusters)
# print(new_X)
# print(new_Y)

import os

# 假设你的脚本和这些文件在同一个目录下
# 如果不是，请修改下面的路径为你的文件所在的目录
directory = 'train_pth/'

# 遍历指定目录
for filename in os.listdir(directory):
    # 检查文件名是否符合模式
    if filename.startswith('model_state_dict_train_gru_CTTS_') and filename.endswith('.pth'):
        # 提取文件名中的数字部分
        try:
            num_str = filename.split('_')[-1].split('.')[0]  # 分割'_'后取最后一个部分，再分割'.'取第一个部分
            num = int(num_str)

            # 判断数字是否在删除范围内
            if 1 <= num <= 24 or 36 <= num <= 200:
                # 构造完整的文件路径
                file_path = os.path.join(directory, filename)
                # 删除文件
                os.remove(file_path)
                print(f'Deleted: {file_path}')
        except ValueError:
            # 如果文件名不符合预期的格式（例如，没有数字或数字部分无法转换为整数），则忽略
            continue

print('Done.')