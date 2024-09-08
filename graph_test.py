import bert_data
import graph_model
import bert_model
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertModel,BertTokenizer
import torch.nn.functional as F
import os
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import DataLoader
import dgl

model_name = './bert-base-Chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)

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
x_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))
device = 'cpu'
mymodel = graph_model.HeteroNodeClassifier(hidden_size=10, num_classes=1).to(device)
mymodel.load_state_dict(torch.load('model_graph.pth'))
def get_Data(corporate_name):
    df = pd.DataFrame(columns=['Time', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    news_path = 'corporates_news/' + corporate_name + '.xlsx'
    news_df = pd.read_excel(news_path)
    news_df['datetime'] = pd.to_datetime(news_df['date']).dt.floor('d')
    policy_path = 'policy_tensor.xlsx'
    policy_df = pd.read_excel(policy_path)

    policy_df['时间'] = pd.to_datetime(policy_df['时间']).dt.to_period('M')
    # print(policy_df)

    file_path = 'stock_data/2022-1/训练集/A股/' + corporate_name + '.xlsx'
    if os.path.exists(file_path):
        data_df = pd.read_excel(file_path)
    else:
        file_path = 'stock_data/2022-1/训练集/港股/' + corporate_name + '.xlsx'
        data_df = pd.read_excel(file_path)
    data_df['date'] = pd.to_datetime(data_df['date']).dt.floor('d')
    data_tmp = data_df.iloc[:, 1:58].values
    data_df.iloc[:, 1:58] = x_scaler.fit_transform(data_tmp)
    data_tmp_0 = data_df.iloc[:, 58:59].values
    condition = data_tmp_0 >= 0.5
    # 对满足条件的数进行操作
    data_tmp_0[condition] = (data_tmp_0[condition] - 0.5) / 2 + 0.5
    # 对不满足条件的数进行操作
    data_tmp_0[~condition] = 0.5 - (0.5 - data_tmp_0[~condition]) / 2
    data_df.iloc[:, 58:59] = data_tmp_0
    for i in range(0, len(data_df)):
        rcd_n = 1
        rcd_p = 1
        # print(i)
        '''
        先测试小数据，原数据为486
        '''
        # current_df = data_df.iloc[i, :]
        # print(current_df)
        # print(i)
        stock_df = data_df.iloc[i, 1:6].values
        financial_df = data_df.iloc[i, 6:22].values
        macro_df = data_df.iloc[i, 22:43].values
        label_df = data_df.iloc[i, 58]
        # print(label_df)
        stock_df = pd.DataFrame(stock_df, dtype=np.float32)
        financial_df = pd.DataFrame(financial_df, dtype=np.float32)
        # label_df = pd.DataFrame(label_df, dtype=np.float32)
        macro_df = pd.DataFrame(macro_df, dtype=np.float32)

        stock_df = torch.tensor(stock_df.values)
        financial_df = torch.tensor(financial_df.values)
        macro_df = torch.tensor(macro_df.values)
        # label_df = torch.tensor(label_df.values)
        stock_df = stock_df.view(1, -1)
        financial_df = financial_df.view(1, -1)
        macro_df = macro_df.view(1, -1)
        label_df = label_df
        # print(stock_df.shape)
        target_time = data_df.iloc[i, 0]
        # 找到指定时间的所有数据
        selected_data = news_df[news_df['date'] == target_time]
        # 检查是否找到了数据
        if selected_data.empty:
            # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
            rcd_n = 0
            selected_data = pd.DataFrame(columns=news_df.columns, data=[[0] * len(news_df.columns)])

        selected_df = selected_data.iloc[:, 0:768].values

        if selected_data.shape[1] < 768:
            selected_df = np.zeros((selected_data.shape[0], 768))  # 创建一个全零数组，形状为 (num_samples, 768)

        selected_df = pd.DataFrame(selected_df, dtype=np.float32)
        selected_df = torch.tensor(selected_df.values)
        selected_df = selected_df.view(1, -1)

        # 将数据放入数组中
        # selected_data_array = selected_df.values.tolist()
        stock_df = stock_df.view(1, -1)
        financial_df = financial_df.view(1, -1)
        macro_df = macro_df.view(1, -1)

        target_time_0 = pd.to_datetime(target_time).to_period('M')

        selected_policy = policy_df[policy_df['时间'] == target_time_0]
        # 检查是否找到了数据
        if selected_policy.empty:
            rcd_p = 0
            # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
            selected_policy = pd.DataFrame(columns=policy_df.columns, data=[[0] * len(policy_df.columns)])

        selected_df_p = selected_policy.iloc[:, 0:768].values

        if selected_policy.shape[1] < 768:
            selected_df_p = np.zeros((selected_policy.shape[0], 768))  # 创建一个全零数组，形状为 (num_samples, 768)

        selected_df_p = pd.DataFrame(selected_df_p, dtype=np.float32)
        selected_df_p = torch.tensor(selected_df_p.values)
        selected_df_p = selected_df_p.view(1, -1)
        record = rcd_p * 10 + rcd_n * 1
        if record == 11:
            title_list = ['stock', 'financial', 'macro', 'news', 'policy']
            G = dgl.heterograph({
                ('stock', 'stock_financial', 'financial'): ([0], [0]),
                ('stock', 'stock_macro', 'macro'): ([0], [0]),
                ('stock', 'stock_news', 'news'): ([0], [0]),
                ('stock', 'stock_policy', 'policy'): ([0], [0]),
                ('financial', 'financial_macro', 'macro'): ([0], [0]),
                ('financial', 'financial_stock', 'stock'): ([0], [0]),
                ('financial', 'financial_news', 'news'): ([0], [0]),
                ('financial', 'financial_policy', 'policy'): ([0], [0]),
                ('macro', 'macro_stock', 'stock'): ([0], [0]),
                ('macro', 'macro_financial', 'financial'): ([0], [0]),
                ('macro', 'macro_news', 'news'): ([0], [0]),
                ('macro', 'macro_policy', 'policy'): ([0], [0]),
                ('news', 'news_stock', 'stock'): ([0], [0]),
                ('news', 'news_financial', 'financial'): ([0], [0]),
                ('news', 'news_macro', 'macro'): ([0], [0]),
                ('news', 'news_policy', 'policy'): ([0], [0]),
                ('policy', 'policy_stock', 'stock'): ([0], [0]),
                ('policy', 'policy_macro', 'macro'): ([0], [0]),
                ('policy', 'policy_news', 'news'): ([0], [0]),
                ('policy', 'policy_financial', 'financial'): ([0], [0])
            })
            # 初始化节点特征
            G.nodes['stock'].data['feat'] = stock_df
            G.nodes['financial'].data['feat'] = financial_df
            G.nodes['macro'].data['feat'] = macro_df
            G.nodes['news'].data['feat'] = selected_df
            G.nodes['policy'].data['feat'] = selected_df_p
            # 初始化边特征
            for tl1 in title_list:
                for tl2 in title_list:
                    if tl1 != tl2:
                        edge = tl1 + '_' + tl2
                        G.edges[edge].data['feat'] = torch.zeros(1, 10)
            graph_dict = {'stock': G.nodes['stock'].data['feat'],
                          'financial': G.nodes['financial'].data['feat'],
                          'macro': G.nodes['macro'].data['feat'],
                          'news': G.nodes['news'].data['feat'],
                          'policy': G.nodes['policy'].data['feat']}
        elif record == 10:
            title_list = ['stock', 'financial', 'macro', 'policy']
            G = dgl.heterograph({
                ('stock', 'stock_financial', 'financial'): ([0], [0]),
                ('stock', 'stock_macro', 'macro'): ([0], [0]),
                # ('stock', 'stock_news', 'news'): ([0], [0]),
                ('stock', 'stock_policy', 'policy'): ([0], [0]),
                ('financial', 'financial_macro', 'macro'): ([0], [0]),
                ('financial', 'financial_stock', 'stock'): ([0], [0]),
                # ('financial', 'financial_news', 'news'): ([0], [0]),
                ('financial', 'financial_policy', 'policy'): ([0], [0]),
                ('macro', 'macro_stock', 'stock'): ([0], [0]),
                ('macro', 'macro_financial', 'financial'): ([0], [0]),
                # ('macro', 'macro_news', 'news'): ([0], [0]),
                ('macro', 'macro_policy', 'policy'): ([0], [0]),
                # ('news', 'news_stock', 'stock'): ([0], [0]),
                # ('news', 'news_financial', 'financial'): ([0], [0]),
                # ('news', 'news_macro', 'macro'): ([0], [0]),
                # ('news', 'news_policy', 'policy'): ([0], [0]),
                ('policy', 'policy_stock', 'stock'): ([0], [0]),
                ('policy', 'policy_macro', 'macro'): ([0], [0]),
                # ('policy', 'policy_news', 'news'): ([0], [0]),
                ('policy', 'policy_financial', 'financial'): ([0], [0])
            })
            # 初始化节点特征
            G.nodes['stock'].data['feat'] = stock_df
            G.nodes['financial'].data['feat'] = financial_df
            G.nodes['macro'].data['feat'] = macro_df
            # G.nodes['news'].data['feat'] = selected_df
            G.nodes['policy'].data['feat'] = selected_df_p
            # 初始化边特征
            for tl1 in title_list:
                for tl2 in title_list:
                    if tl1 != tl2:
                        edge = tl1 + '_' + tl2
                        G.edges[edge].data['feat'] = torch.zeros(1, 10)
            graph_dict = {'stock': G.nodes['stock'].data['feat'],
                          'financial': G.nodes['financial'].data['feat'],
                          'macro': G.nodes['macro'].data['feat'],
                          'policy': G.nodes['policy'].data['feat']}
        elif record == 1:
            title_list = ['stock', 'financial', 'macro', 'news']
            G = dgl.heterograph({
                ('stock', 'stock_financial', 'financial'): ([0], [0]),
                ('stock', 'stock_macro', 'macro'): ([0], [0]),
                ('stock', 'stock_news', 'news'): ([0], [0]),
                # ('stock', 'stock_policy', 'policy'): ([0], [0]),
                ('financial', 'financial_macro', 'macro'): ([0], [0]),
                ('financial', 'financial_stock', 'stock'): ([0], [0]),
                ('financial', 'financial_news', 'news'): ([0], [0]),
                # ('financial', 'financial_policy', 'policy'): ([0], [0]),
                ('macro', 'macro_stock', 'stock'): ([0], [0]),
                ('macro', 'macro_financial', 'financial'): ([0], [0]),
                ('macro', 'macro_news', 'news'): ([0], [0]),
                # ('macro', 'macro_policy', 'policy'): ([0], [0]),
                ('news', 'news_stock', 'stock'): ([0], [0]),
                ('news', 'news_financial', 'financial'): ([0], [0]),
                ('news', 'news_macro', 'macro'): ([0], [0]),
                # ('news', 'news_policy', 'policy'): ([0], [0]),
                # ('policy', 'policy_stock', 'stock'): ([0], [0]),
                # ('policy', 'policy_macro', 'macro'): ([0], [0]),
                # ('policy', 'policy_news', 'news'): ([0], [0]),
                # ('policy', 'policy_financial', 'financial'): ([0], [0])
            })
            # 初始化节点特征
            G.nodes['stock'].data['feat'] = stock_df
            G.nodes['financial'].data['feat'] = financial_df
            G.nodes['macro'].data['feat'] = macro_df
            G.nodes['news'].data['feat'] = selected_df
            # G.nodes['policy'].data['feat'] = selected_df_p
            # 初始化边特征
            for tl1 in title_list:
                for tl2 in title_list:
                    if tl1 != tl2:
                        edge = tl1 + '_' + tl2
                        G.edges[edge].data['feat'] = torch.zeros(1, 10)
            graph_dict = {'stock': G.nodes['stock'].data['feat'],
                          'financial': G.nodes['financial'].data['feat'],
                          'macro': G.nodes['macro'].data['feat'],
                          'news': G.nodes['news'].data['feat']}
        else:
            title_list = ['stock', 'financial', 'macro']
            G = dgl.heterograph({
                ('stock', 'stock_financial', 'financial'): ([0], [0]),
                ('stock', 'stock_macro', 'macro'): ([0], [0]),
                # ('stock', 'stock_news', 'news'): ([0], [0]),
                # ('stock', 'stock_policy', 'policy'): ([0], [0]),
                ('financial', 'financial_macro', 'macro'): ([0], [0]),
                ('financial', 'financial_stock', 'stock'): ([0], [0]),
                # ('financial', 'financial_news', 'news'): ([0], [0]),
                # ('financial', 'financial_policy', 'policy'): ([0], [0]),
                ('macro', 'macro_stock', 'stock'): ([0], [0]),
                ('macro', 'macro_financial', 'financial'): ([0], [0]),
                # ('macro', 'macro_news', 'news'): ([0], [0]),
                # ('macro', 'macro_policy', 'policy'): ([0], [0]),
                # ('news', 'news_stock', 'stock'): ([0], [0]),
                # ('news', 'news_financial', 'financial'): ([0], [0]),
                # ('news', 'news_macro', 'macro'): ([0], [0]),
                # ('news', 'news_policy', 'policy'): ([0], [0]),
                # ('policy', 'policy_stock', 'stock'): ([0], [0]),
                # ('policy', 'policy_macro', 'macro'): ([0], [0]),
                # ('policy', 'policy_news', 'news'): ([0], [0]),
                # ('policy', 'policy_financial', 'financial'): ([0], [0])
            })
            # 初始化节点特征
            G.nodes['stock'].data['feat'] = stock_df
            G.nodes['financial'].data['feat'] = financial_df
            G.nodes['macro'].data['feat'] = macro_df
            # G.nodes['news'].data['feat'] = selected_df
            # G.nodes['policy'].data['feat'] = selected_df_p
            # 初始化边特征
            for tl1 in title_list:
                for tl2 in title_list:
                    if tl1 != tl2:
                        edge = tl1 + '_' + tl2
                        G.edges[edge].data['feat'] = torch.zeros(1, 10)
            graph_dict = {'stock': G.nodes['stock'].data['feat'],
                          'financial': G.nodes['financial'].data['feat'],
                          'macro': G.nodes['macro'].data['feat']}

        _, output = mymodel(G, graph_dict)
        tensor_values = output.cpu().detach().numpy().flatten().tolist()
        row_data = [target_time] + tensor_values

        df.loc[len(df)] = row_data

    df.to_excel('graph_daily_G/' + corporate_name + '.xlsx', index=False)
    return

for corporate in corporate_list_0:
    get_Data(corporate_name=corporate)