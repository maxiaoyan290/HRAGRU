import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

#
# train 2022-1
# test 2022-4从2021年末开始

corporate_list = ['万业企业']

corporate_list_0 = ['万业企业', '万方发展', '万科A', '三湘股份', '上实发展', '上海临港',
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
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

class trainDataset():
    def __init__(self):
        number_to_name = {
            1: "Stock",
            2: "Financial",
            3: "Macro",
            4: "News",
            5: "Policy"
        }
        self.ntn = number_to_name
        self.stride = 30
        '''
        先测试小数据，原数据为30
        '''

        data = []
        wobert_data = []
        wognn_data = []
        graph_data = []
        news = []
        mask_data = []
        corporate_data = []
        corporate_mask_data = []
        policies_data = []
        policy_masks_data = []
        label = []

        for corporate in corporate_list:
            print(corporate)
            return_list, label_lists, news_list, policy_lists, wobert_lists, wognn_lists = self.get_Data(corporate)
            data_list, label_list, new_list, policy_list, wobert_list, wognn_list = \
                self.slide_windows(return_list, label_lists, news_list, policy_lists, wobert_lists, wognn_lists)
            data = data + data_list
            label = label + label_list
            news = news + new_list
            policies_data = policies_data + policy_list
            wobert_data = wobert_data + wobert_list
            wognn_data = wognn_data + wognn_list
        length = len(data)

        self.data = data
        self.graph_data = graph_data
        self.new_data = news
        self.mask_data = mask_data
        self.corporate_data = corporate_data
        self.corporate_mask_data = corporate_mask_data
        self.policies_data = policies_data
        self.policy_masks_data = policy_masks_data
        self.label = label
        self.len = length
        self.wobert_data = wobert_data
        self.wognn_data = wognn_data

    def get_Data(self, corporate_name):
        return_list = []
        label_list = []
        news_list = []
        policy_list = []
        without_bert_list = []
        without_gnn_list = []

        news_path = 'corporates_news/' + corporate_name + '.xlsx'
        news_df = pd.read_excel(news_path)
        news_df['datetime'] = pd.to_datetime(news_df['date']).dt.floor('d')
        policy_path = 'policy_tensor.xlsx'
        policy_df = pd.read_excel(policy_path)

        policy_df['时间'] = pd.to_datetime(policy_df['时间']).dt.to_period('M')
        #print(policy_df)

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

        bert_path = 'corporates_daily_train/' + corporate_name + '.xlsx'
        bert_df = pd.read_excel(bert_path)
        bert_df['Time'] = pd.to_datetime(bert_df['Time']).dt.floor('d')
        bert_tmp = bert_df.iloc[:, 1:21].values
        bert_df.iloc[:, 1:21] = x_scaler.fit_transform(bert_tmp)
        graph_path = 'graph_daily_G_train/' + corporate_name + '.xlsx'
        graph_df = pd.read_excel(graph_path)
        graph_df['Time'] = pd.to_datetime(graph_df['Time']).dt.floor('d')
        graph_tmp = graph_df.iloc[:, 1:11].values
        graph_df.iloc[:, 1:11] = x_scaler.fit_transform(graph_tmp)
        for i in range(0, len(data_df)):
            # print(i)
            '''
            先测试小数据，原数据为486
            '''
            # current_df = data_df.iloc[i, :]
            # print(current_df)
            #print(i)
            stock_df = data_df.iloc[i, 1:6]
            financial_df = data_df.iloc[i, 6:22]
            macro_df = data_df.iloc[i, 22:43]
            test_df = data_df.iloc[i, 1:58].values
            label_df = data_df.iloc[i, 58:59]
            #print(label_df)
            stock_df = pd.DataFrame(stock_df, dtype=np.float32)
            financial_df = pd.DataFrame(financial_df, dtype=np.float32)
            label_df = pd.DataFrame(label_df, dtype=np.float32)
            macro_df = pd.DataFrame(macro_df, dtype=np.float32)
            test_df = pd.DataFrame(test_df, dtype=np.float32)

            stock_df = torch.tensor(stock_df.values)
            financial_df = torch.tensor(financial_df.values)
            macro_df = torch.tensor(macro_df.values)
            label_df = torch.tensor(label_df.values)
            test_df = torch.tensor(test_df.values)
            stock_df = stock_df.view(1, -1)
            financial_df = financial_df.view(1, -1)
            macro_df = macro_df.view(1, -1)
            label_df = label_df.view(1, -1)
            test_df = test_df.view(1, -1)
            #print(stock_df.shape)
            target_time = data_df.iloc[i, 0]
            # 找到指定时间的所有数据
            selected_data = news_df[news_df['date'] == target_time]
            # 检查是否找到了数据
            if selected_data.empty:
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                selected_data = pd.DataFrame(columns=news_df.columns, data=[[0]*len(news_df.columns)])

            selected_df = selected_data.iloc[:, 0:768].values

            if selected_data.shape[1] < 768:
                selected_df = np.zeros((selected_data.shape[0], 768))  # 创建一个全零数组，形状为 (num_samples, 768)

            selected_df = pd.DataFrame(selected_df, dtype=np.float32)
            selected_df = torch.tensor(selected_df.values)
            selected_df = selected_df.view(1, -1)

            news_list.append(selected_df)

            selected_bert = bert_df[bert_df['Time'] == target_time]
            # 检查是否找到了数据
            if selected_bert.empty:
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                selected_bert = pd.DataFrame(columns=bert_df.columns, data=[[0]*len(bert_df.columns)])

            selected_df_b = selected_bert.iloc[:, 1:21].values

            if selected_bert.shape[1] < 20:
                selected_df_b = np.zeros((selected_bert.shape[0], 20))  # 创建一个全零数组，形状为 (num_samples, 768)

            selected_df_b = pd.DataFrame(selected_df_b, dtype=np.float32)
            selected_df_b = torch.tensor(selected_df_b.values)
            selected_df_b = selected_df_b.view(1, -1)

            selected_graph = graph_df[graph_df['Time'] == target_time]
            # 检查是否找到了数据
            if selected_graph.empty:
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                selected_graph = pd.DataFrame(columns=graph_df.columns, data=[[0]*len(graph_df.columns)])

            selected_df_g = selected_graph.iloc[:, 1:11].values

            if selected_graph.shape[1] < 10:
                selected_df_g = np.zeros((selected_graph.shape[0], 10))  # 创建一个全零数组，形状为 (num_samples, 768)

            selected_df_g = pd.DataFrame(selected_df_g, dtype=np.float32)
            selected_df_g = torch.tensor(selected_df_g.values)
            selected_df_g = selected_df_g.view(1, -1)

            target_time = pd.to_datetime(target_time).to_period('M')

            selected_policy = policy_df[policy_df['时间'] == target_time]
            # 检查是否找到了数据
            if selected_policy.empty:
                rcd_p = 0
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                selected_policy = pd.DataFrame(columns=policy_df.columns, data=[[0]*len(policy_df.columns)])

            selected_df_p = selected_policy.iloc[:, 0:768].values

            if selected_policy.shape[1] < 768:
                selected_df_p = np.zeros((selected_policy.shape[0], 768))  # 创建一个全零数组，形状为 (num_samples, 768)

            selected_df_p = pd.DataFrame(selected_df_p, dtype=np.float32)
            selected_df_p = torch.tensor(selected_df_p.values)
            selected_df_p = selected_df_p.view(1, -1)
            policy_list.append(selected_df_p)

            # 将数据放入数组中
            # selected_data_array = selected_df.values.tolist()
            stock_df = stock_df.view(1, -1)
            financial_df = financial_df.view(1, -1)
            macro_df = macro_df.view(1, -1)
            test_df = test_df.view(1, -1)
            # print("stock:", stock_df.shape)
            # print("financial:", financial_df.shape)
            # print("macro:", macro_df.shape)
            total_df = torch.cat((test_df, selected_df_b, selected_df_g), dim=1)
            wobert_df = torch.cat((test_df, selected_df_g), dim=1)
            wognn_df = torch.cat((test_df, selected_df_b), dim=1)

            return_list.append(total_df)
            # print(graph_tensor.shape)
            # print(masked_graph_tensor_1.shape)
            label_list.append(label_df)
            without_bert_list.append(wobert_df)
            without_gnn_list.append(wognn_df)
            # print(len(return_list))

        return return_list, label_list, news_list, policy_list, without_bert_list, without_gnn_list

    def slide_windows(self, return_list, label_lists, news_lists, policy_list, without_bert_list, without_gnn_list):
        data = []
        label = []
        news = []
        policies = []
        wobert = []
        wognn = []
        for i in range(0, len(return_list) - self.stride - 1):
            #print(return_list[i:i + self.stride])
            return_df = pd.DataFrame([tensor.flatten().numpy() for tensor in return_list[i:i + self.stride]], dtype=np.float32)
            return_df = torch.tensor(return_df.values)
            data.append(return_df)
            label_df = pd.DataFrame([tensor.flatten().numpy() for tensor in label_lists[i + self.stride]], dtype=np.float32)
            label_df = torch.tensor(label_df.values)
            label.append(label_df)
            news_df = pd.DataFrame([tensor.flatten().numpy() for tensor in news_lists[i:i + self.stride]], dtype=np.float32)
            news_df = torch.tensor(news_df.values)
            news.append(news_df)
            policy_df = pd.DataFrame([tensor.flatten().numpy() for tensor in policy_list[i:i + self.stride]], dtype=np.float32)
            policy_df = torch.tensor(policy_df.values)
            policies.append(policy_df)
            wobert_df = pd.DataFrame([tensor.flatten().numpy() for tensor in without_bert_list[i:i + self.stride]], dtype=np.float32)
            wobert_df = torch.tensor(wobert_df.values)
            wobert.append(wobert_df)
            wognn_df = pd.DataFrame([tensor.flatten().numpy() for tensor in without_gnn_list[i:i + self.stride]], dtype=np.float32)
            wognn_df = torch.tensor(wognn_df.values)
            wognn.append(wognn_df)
        return data, label, news, policies, wobert, wognn

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.new_data[index], self.policies_data[index], self.wobert_data[index], self.wognn_data[index]

    def __len__(self):
        return self.len

class valDataset():
    def __init__(self, corporate):
        number_to_name = {
            1: "Stock",
            2: "Financial",
            3: "Macro",
            4: "News",
            5: "Policy"
        }
        self.ntn = number_to_name
        self.stride = 30
        '''
        先测试小数据，原数据为30
        '''

        data = []
        wobert_data = []
        wognn_data = []
        graph_data = []
        news = []
        mask_data = []
        corporate_data = []
        corporate_mask_data = []
        policies_data = []
        policy_masks_data = []
        label = []

        print(corporate)
        return_list, label_lists, news_list, policy_lists, wobert_lists, wognn_lists = self.get_Data(corporate)
        data_list, label_list, new_list, policy_list, wobert_list, wognn_list = \
            self.slide_windows(return_list, label_lists, news_list, policy_lists, wobert_lists, wognn_lists)
        data = data + data_list
        label = label + label_list
        news = news + new_list
        policies_data = policies_data + policy_list
        wobert_data = wobert_data + wobert_list
        wognn_data = wognn_data + wognn_list
        length = len(data)

        self.data = data
        self.graph_data = graph_data
        self.new_data = news
        self.mask_data = mask_data
        self.corporate_data = corporate_data
        self.corporate_mask_data = corporate_mask_data
        self.policies_data = policies_data
        self.policy_masks_data = policy_masks_data
        self.label = label
        self.len = length
        self.wobert_data = wobert_data
        self.wognn_data = wognn_data

    def get_Data(self, corporate_name):
        return_list = []
        label_list = []
        news_list = []
        policy_list = []
        without_bert_list = []
        without_gnn_list = []

        news_path = 'corporates_news/' + corporate_name + '.xlsx'
        news_df = pd.read_excel(news_path)
        news_df['datetime'] = pd.to_datetime(news_df['date']).dt.floor('d')
        policy_path = 'policy_tensor.xlsx'
        policy_df = pd.read_excel(policy_path)

        policy_df['时间'] = pd.to_datetime(policy_df['时间']).dt.to_period('M')
        #print(policy_df)

        file_path = 'stock_data/2022-4/测试集/A股/' + corporate_name + '.xlsx'
        if os.path.exists(file_path):
            data_df = pd.read_excel(file_path)
        else:
            file_path = 'stock_data/2022-4/测试集/港股/' + corporate_name + '.xlsx'
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

        bert_path = 'corporates_daily_test/' + corporate_name + '.xlsx'
        bert_df = pd.read_excel(bert_path)
        bert_df['Time'] = pd.to_datetime(bert_df['Time']).dt.floor('d')
        bert_tmp = bert_df.iloc[:, 1:21].values
        bert_df.iloc[:, 1:21] = x_scaler.fit_transform(bert_tmp)
        graph_path = 'graph_daily_G_test/' + corporate_name + '.xlsx'
        graph_df = pd.read_excel(graph_path)
        graph_df['Time'] = pd.to_datetime(graph_df['Time']).dt.floor('d')
        graph_tmp = graph_df.iloc[:, 1:11].values
        graph_df.iloc[:, 1:11] = x_scaler.fit_transform(graph_tmp)
        for i in range(0, len(data_df)):
            # print(i)
            '''
            先测试小数据，原数据为486
            '''
            # current_df = data_df.iloc[i, :]
            # print(current_df)
            #print(i)
            stock_df = data_df.iloc[i, 1:6]
            financial_df = data_df.iloc[i, 6:22]
            macro_df = data_df.iloc[i, 22:43]
            test_df = data_df.iloc[i, 1:58].values
            label_df = data_df.iloc[i, 58:59]
            #print(label_df)
            stock_df = pd.DataFrame(stock_df, dtype=np.float32)
            financial_df = pd.DataFrame(financial_df, dtype=np.float32)
            label_df = pd.DataFrame(label_df, dtype=np.float32)
            macro_df = pd.DataFrame(macro_df, dtype=np.float32)
            test_df = pd.DataFrame(test_df, dtype=np.float32)

            stock_df = torch.tensor(stock_df.values)
            financial_df = torch.tensor(financial_df.values)
            macro_df = torch.tensor(macro_df.values)
            label_df = torch.tensor(label_df.values)
            test_df = torch.tensor(test_df.values)
            stock_df = stock_df.view(1, -1)
            financial_df = financial_df.view(1, -1)
            macro_df = macro_df.view(1, -1)
            label_df = label_df.view(1, -1)
            test_df = test_df.view(1, -1)
            #print(stock_df.shape)
            target_time = data_df.iloc[i, 0]
            # 找到指定时间的所有数据
            selected_data = news_df[news_df['date'] == target_time]
            # 检查是否找到了数据
            if selected_data.empty:
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                selected_data = pd.DataFrame(columns=news_df.columns, data=[[0]*len(news_df.columns)])

            selected_df = selected_data.iloc[:, 0:768].values

            if selected_data.shape[1] < 768:
                selected_df = np.zeros((selected_data.shape[0], 768))  # 创建一个全零数组，形状为 (num_samples, 768)

            selected_df = pd.DataFrame(selected_df, dtype=np.float32)
            selected_df = torch.tensor(selected_df.values)
            selected_df = selected_df.view(1, -1)

            news_list.append(selected_df)

            selected_bert = bert_df[bert_df['Time'] == target_time]
            # 检查是否找到了数据
            if selected_bert.empty:
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                selected_bert = pd.DataFrame(columns=bert_df.columns, data=[[0]*len(bert_df.columns)])

            selected_df_b = selected_bert.iloc[:, 1:21].values

            if selected_bert.shape[1] < 20:
                selected_df_b = np.zeros((selected_bert.shape[0], 20))  # 创建一个全零数组，形状为 (num_samples, 768)

            selected_df_b = pd.DataFrame(selected_df_b, dtype=np.float32)
            selected_df_b = torch.tensor(selected_df_b.values)
            selected_df_b = selected_df_b.view(1, -1)

            selected_graph = graph_df[graph_df['Time'] == target_time]
            # 检查是否找到了数据
            if selected_graph.empty:
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                selected_graph = pd.DataFrame(columns=graph_df.columns, data=[[0]*len(graph_df.columns)])

            selected_df_g = selected_graph.iloc[:, 1:11].values

            if selected_graph.shape[1] < 10:
                selected_df_g = np.zeros((selected_graph.shape[0], 10))  # 创建一个全零数组，形状为 (num_samples, 768)

            selected_df_g = pd.DataFrame(selected_df_g, dtype=np.float32)
            selected_df_g = torch.tensor(selected_df_g.values)
            selected_df_g = selected_df_g.view(1, -1)

            target_time = pd.to_datetime(target_time).to_period('M')

            selected_policy = policy_df[policy_df['时间'] == target_time]
            # 检查是否找到了数据
            if selected_policy.empty:
                rcd_p = 0
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                selected_policy = pd.DataFrame(columns=policy_df.columns, data=[[0]*len(policy_df.columns)])

            selected_df_p = selected_policy.iloc[:, 0:768].values

            if selected_policy.shape[1] < 768:
                selected_df_p = np.zeros((selected_policy.shape[0], 768))  # 创建一个全零数组，形状为 (num_samples, 768)

            selected_df_p = pd.DataFrame(selected_df_p, dtype=np.float32)
            selected_df_p = torch.tensor(selected_df_p.values)
            selected_df_p = selected_df_p.view(1, -1)
            policy_list.append(selected_df_p)

            # 将数据放入数组中
            # selected_data_array = selected_df.values.tolist()
            stock_df = stock_df.view(1, -1)
            financial_df = financial_df.view(1, -1)
            macro_df = macro_df.view(1, -1)
            test_df = test_df.view(1, -1)
            # print("stock:", stock_df.shape)
            # print("financial:", financial_df.shape)
            # print("macro:", macro_df.shape)
            total_df = torch.cat((test_df, selected_df_b, selected_df_g), dim=1)
            wobert_df = torch.cat((test_df, selected_df_g), dim=1)
            wognn_df = torch.cat((test_df, selected_df_b), dim=1)

            return_list.append(total_df)
            # print(graph_tensor.shape)
            # print(masked_graph_tensor_1.shape)
            label_list.append(label_df)
            without_bert_list.append(wobert_df)
            without_gnn_list.append(wognn_df)
            # print(len(return_list))

        return return_list, label_list, news_list, policy_list, without_bert_list, without_gnn_list

    def slide_windows(self, return_list, label_lists, news_lists, policy_list, without_bert_list, without_gnn_list):
        data = []
        label = []
        news = []
        policies = []
        wobert = []
        wognn = []
        for i in range(0, len(return_list) - self.stride - 1):
            #print(return_list[i:i + self.stride])
            return_df = pd.DataFrame([tensor.flatten().numpy() for tensor in return_list[i:i + self.stride]], dtype=np.float32)
            return_df = torch.tensor(return_df.values)
            data.append(return_df)
            label_df = pd.DataFrame([tensor.flatten().numpy() for tensor in label_lists[i + self.stride]], dtype=np.float32)
            label_df = torch.tensor(label_df.values)
            label.append(label_df)
            news_df = pd.DataFrame([tensor.flatten().numpy() for tensor in news_lists[i:i + self.stride]], dtype=np.float32)
            news_df = torch.tensor(news_df.values)
            news.append(news_df)
            policy_df = pd.DataFrame([tensor.flatten().numpy() for tensor in policy_list[i:i + self.stride]], dtype=np.float32)
            policy_df = torch.tensor(policy_df.values)
            policies.append(policy_df)
            wobert_df = pd.DataFrame([tensor.flatten().numpy() for tensor in without_bert_list[i:i + self.stride]], dtype=np.float32)
            wobert_df = torch.tensor(wobert_df.values)
            wobert.append(wobert_df)
            wognn_df = pd.DataFrame([tensor.flatten().numpy() for tensor in without_gnn_list[i:i + self.stride]], dtype=np.float32)
            wognn_df = torch.tensor(wognn_df.values)
            wognn.append(wognn_df)
        return data, label, news, policies, wobert, wognn

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.new_data[index], self.policies_data[index], self.wobert_data[index], self.wognn_data[index]

    def __len__(self):
        return self.len

class testDataset():
    def __init__(self):
        number_to_name = {
            1: "Stock",
            2: "Financial",
            3: "Macro",
            4: "News",
            5: "Policy"
        }
        self.ntn = number_to_name
        self.stride = 30
        '''
        先测试小数据，原数据为30
        '''

        data = []
        wobert_data = []
        wognn_data = []
        graph_data = []
        news = []
        mask_data = []
        corporate_data = []
        corporate_mask_data = []
        policies_data = []
        policy_masks_data = []
        label = []

        for corporate in corporate_list:
            print(corporate)
            return_list, label_lists, news_list, policy_lists, wobert_lists, wognn_lists = self.get_Data(corporate)
            data_list, label_list, new_list, policy_list, wobert_list, wognn_list = \
                self.slide_windows(return_list, label_lists, news_list, policy_lists, wobert_lists, wognn_lists)
            data = data + data_list
            label = label + label_list
            news = news + new_list
            policies_data = policies_data + policy_list
            wobert_data = wobert_data + wobert_list
            wognn_data = wognn_data + wognn_list
        length = len(data)

        self.data = data
        self.graph_data = graph_data
        self.new_data = news
        self.mask_data = mask_data
        self.corporate_data = corporate_data
        self.corporate_mask_data = corporate_mask_data
        self.policies_data = policies_data
        self.policy_masks_data = policy_masks_data
        self.label = label
        self.len = length
        self.wobert_data = wobert_data
        self.wognn_data = wognn_data

    def get_Data(self, corporate_name):
        return_list = []
        label_list = []
        news_list = []
        policy_list = []
        without_bert_list = []
        without_gnn_list = []

        news_path = 'corporates_news/' + corporate_name + '.xlsx'
        news_df = pd.read_excel(news_path)
        news_df['datetime'] = pd.to_datetime(news_df['date']).dt.floor('d')
        policy_path = 'policy_tensor.xlsx'
        policy_df = pd.read_excel(policy_path)

        policy_df['时间'] = pd.to_datetime(policy_df['时间']).dt.to_period('M')
        #print(policy_df)

        file_path = 'stock_data/2022-4/测试集/A股/' + corporate_name + '.xlsx'
        if os.path.exists(file_path):
            data_df = pd.read_excel(file_path)
        else:
            file_path = 'stock_data/2022-4/测试集/港股/' + corporate_name + '.xlsx'
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

        bert_path = 'corporates_daily_test/' + corporate_name + '.xlsx'
        bert_df = pd.read_excel(bert_path)
        bert_df['Time'] = pd.to_datetime(bert_df['Time']).dt.floor('d')
        bert_tmp = bert_df.iloc[:, 1:21].values
        bert_df.iloc[:, 1:21] = x_scaler.fit_transform(bert_tmp)
        graph_path = 'graph_daily_G_test/' + corporate_name + '.xlsx'
        graph_df = pd.read_excel(graph_path)
        graph_df['Time'] = pd.to_datetime(graph_df['Time']).dt.floor('d')
        graph_tmp = graph_df.iloc[:, 1:11].values
        graph_df.iloc[:, 1:11] = x_scaler.fit_transform(graph_tmp)
        for i in range(0, len(data_df)):
            # print(i)
            '''
            先测试小数据，原数据为486
            '''
            # current_df = data_df.iloc[i, :]
            # print(current_df)
            #print(i)
            stock_df = data_df.iloc[i, 1:6]
            financial_df = data_df.iloc[i, 6:22]
            macro_df = data_df.iloc[i, 22:43]
            test_df = data_df.iloc[i, 1:58].values
            label_df = data_df.iloc[i, 58:59]
            #print(label_df)
            stock_df = pd.DataFrame(stock_df, dtype=np.float32)
            financial_df = pd.DataFrame(financial_df, dtype=np.float32)
            label_df = pd.DataFrame(label_df, dtype=np.float32)
            macro_df = pd.DataFrame(macro_df, dtype=np.float32)
            test_df = pd.DataFrame(test_df, dtype=np.float32)

            stock_df = torch.tensor(stock_df.values)
            financial_df = torch.tensor(financial_df.values)
            macro_df = torch.tensor(macro_df.values)
            label_df = torch.tensor(label_df.values)
            test_df = torch.tensor(test_df.values)
            stock_df = stock_df.view(1, -1)
            financial_df = financial_df.view(1, -1)
            macro_df = macro_df.view(1, -1)
            label_df = label_df.view(1, -1)
            test_df = test_df.view(1, -1)
            #print(stock_df.shape)
            target_time = data_df.iloc[i, 0]
            # 找到指定时间的所有数据
            selected_data = news_df[news_df['date'] == target_time]
            # 检查是否找到了数据
            if selected_data.empty:
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                selected_data = pd.DataFrame(columns=news_df.columns, data=[[0]*len(news_df.columns)])

            selected_df = selected_data.iloc[:, 0:768].values

            if selected_data.shape[1] < 768:
                selected_df = np.zeros((selected_data.shape[0], 768))  # 创建一个全零数组，形状为 (num_samples, 768)

            selected_df = pd.DataFrame(selected_df, dtype=np.float32)
            selected_df = torch.tensor(selected_df.values)
            selected_df = selected_df.view(1, -1)

            news_list.append(selected_df)

            selected_bert = bert_df[bert_df['Time'] == target_time]
            # 检查是否找到了数据
            if selected_bert.empty:
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                selected_bert = pd.DataFrame(columns=bert_df.columns, data=[[0]*len(bert_df.columns)])

            selected_df_b = selected_bert.iloc[:, 1:21].values

            if selected_bert.shape[1] < 20:
                selected_df_b = np.zeros((selected_bert.shape[0], 20))  # 创建一个全零数组，形状为 (num_samples, 768)

            selected_df_b = pd.DataFrame(selected_df_b, dtype=np.float32)
            selected_df_b = torch.tensor(selected_df_b.values)
            selected_df_b = selected_df_b.view(1, -1)

            selected_graph = graph_df[graph_df['Time'] == target_time]
            # 检查是否找到了数据
            if selected_graph.empty:
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                selected_graph = pd.DataFrame(columns=graph_df.columns, data=[[0]*len(graph_df.columns)])

            selected_df_g = selected_graph.iloc[:, 1:11].values

            if selected_graph.shape[1] < 10:
                selected_df_g = np.zeros((selected_graph.shape[0], 10))  # 创建一个全零数组，形状为 (num_samples, 768)

            selected_df_g = pd.DataFrame(selected_df_g, dtype=np.float32)
            selected_df_g = torch.tensor(selected_df_g.values)
            selected_df_g = selected_df_g.view(1, -1)

            target_time = pd.to_datetime(target_time).to_period('M')

            selected_policy = policy_df[policy_df['时间'] == target_time]
            # 检查是否找到了数据
            if selected_policy.empty:
                rcd_p = 0
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                selected_policy = pd.DataFrame(columns=policy_df.columns, data=[[0]*len(policy_df.columns)])

            selected_df_p = selected_policy.iloc[:, 0:768].values

            if selected_policy.shape[1] < 768:
                selected_df_p = np.zeros((selected_policy.shape[0], 768))  # 创建一个全零数组，形状为 (num_samples, 768)

            selected_df_p = pd.DataFrame(selected_df_p, dtype=np.float32)
            selected_df_p = torch.tensor(selected_df_p.values)
            selected_df_p = selected_df_p.view(1, -1)
            policy_list.append(selected_df_p)

            # 将数据放入数组中
            # selected_data_array = selected_df.values.tolist()
            stock_df = stock_df.view(1, -1)
            financial_df = financial_df.view(1, -1)
            macro_df = macro_df.view(1, -1)
            test_df = test_df.view(1, -1)
            # print("stock:", stock_df.shape)
            # print("financial:", financial_df.shape)
            # print("macro:", macro_df.shape)
            total_df = torch.cat((test_df, selected_df_b, selected_df_g), dim=1)
            wobert_df = torch.cat((test_df, selected_df_g), dim=1)
            wognn_df = torch.cat((test_df, selected_df_b), dim=1)

            return_list.append(total_df)
            # print(graph_tensor.shape)
            # print(masked_graph_tensor_1.shape)
            label_list.append(label_df)
            without_bert_list.append(wobert_df)
            without_gnn_list.append(wognn_df)
            # print(len(return_list))

        return return_list, label_list, news_list, policy_list, without_bert_list, without_gnn_list

    def slide_windows(self, return_list, label_lists, news_lists, policy_list, without_bert_list, without_gnn_list):
        data = []
        label = []
        news = []
        policies = []
        wobert = []
        wognn = []
        for i in range(0, len(return_list) - self.stride - 1):
            #print(return_list[i:i + self.stride])
            return_df = pd.DataFrame([tensor.flatten().numpy() for tensor in return_list[i:i + self.stride]], dtype=np.float32)
            return_df = torch.tensor(return_df.values)
            data.append(return_df)
            label_df = pd.DataFrame([tensor.flatten().numpy() for tensor in label_lists[i + self.stride]], dtype=np.float32)
            label_df = torch.tensor(label_df.values)
            label.append(label_df)
            news_df = pd.DataFrame([tensor.flatten().numpy() for tensor in news_lists[i:i + self.stride]], dtype=np.float32)
            news_df = torch.tensor(news_df.values)
            news.append(news_df)
            policy_df = pd.DataFrame([tensor.flatten().numpy() for tensor in policy_list[i:i + self.stride]], dtype=np.float32)
            policy_df = torch.tensor(policy_df.values)
            policies.append(policy_df)
            wobert_df = pd.DataFrame([tensor.flatten().numpy() for tensor in without_bert_list[i:i + self.stride]], dtype=np.float32)
            wobert_df = torch.tensor(wobert_df.values)
            wobert.append(wobert_df)
            wognn_df = pd.DataFrame([tensor.flatten().numpy() for tensor in without_gnn_list[i:i + self.stride]], dtype=np.float32)
            wognn_df = torch.tensor(wognn_df.values)
            wognn.append(wognn_df)
        return data, label, news, policies, wobert, wognn

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.new_data[index], self.policies_data[index], self.wobert_data[index], self.wognn_data[index]

    def __len__(self):
        return self.len

