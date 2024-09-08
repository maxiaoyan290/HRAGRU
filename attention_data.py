import torch
import numpy as np
import pandas as pd
import os
import dgl
from sklearn.preprocessing import MinMaxScaler
#
# train 2022-1
# test 2022-4从2021年末开始

corporate_list = ['保利地产']

corporate_list_0 = ['万业企业', '万方发展', '万科A', '三湘股份', '上实发展', '上海临港',
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

        stock_data = []
        financial_data = []
        macro_data = []
        news_data = []
        policy_data = []
        label = []

        for corporate in corporate_list:
            print(corporate)
            stock_list, financial_list, macro_list, news_list, policy_list, label_list = self.get_Data(corporate)
            stock_data = stock_data + stock_list
            financial_data = financial_data + financial_list
            macro_data = macro_data + macro_list
            news_data = news_data + news_list
            policy_data = policy_data + policy_list
            label = label + label_list
        length = len(stock_data)

        self.stock_data = stock_data
        self.financial_data = financial_data
        self.macro_data = macro_data
        self.news_data = news_data
        self.policy_data = policy_data
        self.label = label
        self.len = length

    def get_Data(self, corporate_name):
        graph_list = []
        stock_list = []
        financial_list = []
        macro_list = []
        news_list = []
        policy_list = []
        label_list = []

        # 定义填充大小
        padding_size = 512
        padding_len = 10
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
        for i in range(0, len(data_df)):
            rcd_n = 1
            rcd_p = 1
            # print(i)
            '''
            先测试小数据，原数据为486
            '''
            # current_df = data_df.iloc[i, :]
            # print(current_df)
            #print(i)
            stock_df = data_df.iloc[i, 1:6].values
            financial_df = data_df.iloc[i, 6:22].values
            macro_df = data_df.iloc[i, 22:43].values
            label_df = data_df.iloc[i, 58:59]
            #print(label_df)
            stock_df = pd.DataFrame(stock_df, dtype=np.float32)
            financial_df = pd.DataFrame(financial_df, dtype=np.float32)
            label_df = pd.DataFrame(label_df, dtype=np.float32)
            macro_df = pd.DataFrame(macro_df, dtype=np.float32)

            stock_df = torch.tensor(stock_df.values)
            financial_df = torch.tensor(financial_df.values)
            macro_df = torch.tensor(macro_df.values)
            label_df = torch.tensor(label_df.values)
            stock_df = stock_df.view(1, -1)
            financial_df = financial_df.view(1, -1)
            macro_df = macro_df.view(1, -1)
            label_df = label_df.view(1, -1)
            #print(stock_df.shape)
            target_time = data_df.iloc[i, 0]
            # 找到指定时间的所有数据
            selected_data = news_df[news_df['date'] == target_time]
            # 检查是否找到了数据
            if selected_data.empty:
                # 如果没有找到数据，创建一个全零的 DataFrame，并设置列名
                rcd_n = 0
                selected_data = pd.DataFrame(columns=news_df.columns, data=[[0]*len(news_df.columns)])

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
            stock_list.append(stock_df)
            financial_list.append(financial_df)
            macro_list.append(macro_df)
            news_list.append(selected_df)
            policy_list.append(selected_df_p)
            concat_torch = torch.cat((stock_df, financial_df, macro_df, selected_df, selected_df_p), dim=1)
            graph_list.append(concat_torch)
            label_list.append(label_df)
        return stock_list, financial_list, macro_list, news_list, policy_list, label_list

    def __getitem__(self, index):
        return self.stock_data[index], self.financial_data[index], self.macro_data[index], self.news_data[index], self.policy_data[index], self.label[index]

    def __len__(self):
        return self.len