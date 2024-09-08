import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData,Data
import torch.nn.functional as F
import os

#
# train 2022-1
# test 2022-4从2021年末开始

corporate_list_0 = ['保利地产']

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

class trainDataset():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
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

        new_data = []
        mask_data = []
        corporate_data = []
        corporate_mask_data = []
        label = []

        for corporate in corporate_list:
            print(corporate)
            total_news_list, total_masks_list, total_corporates_list, total_corporate_masks_list, label_lists = self.get_Data(corporate)

            new_data = new_data + total_news_list
            mask_data = mask_data + total_masks_list
            corporate_data = corporate_data + total_corporates_list
            corporate_mask_data = corporate_mask_data + total_corporate_masks_list
            label = label + label_lists
        length = len(new_data)

        self.new_data = new_data
        self.mask_data = mask_data
        self.corporate_data = corporate_data
        self.corporate_mask_data = corporate_mask_data
        self.label = label
        self.len = length

    def get_Data(self, corporate_name):
        total_news_list = []
        total_masks_list = []
        total_corporates_list = []
        total_corporate_masks_list = []
        label_list = []

        # 定义填充大小
        padding_size = 512
        news_path = 'news_data/' + corporate_name + '.xlsx'
        news_df = pd.read_excel(news_path)
        news_df['datetime'] = pd.to_datetime(news_df['datetime']).dt.floor('d')
        policy_path = 'policy.xlsx'
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
        for i in range(0, len(data_df)):
            label = data_df.iloc[i, 58].item()
            if label >= 0.5:
                label = 1
            else:
                label = 0
            label_ls = [label]
            label_df = pd.DataFrame(label_ls, dtype=np.float32)
            label_df = torch.tensor(label_df.values)
            target_time = data_df.iloc[i, 0]
            # 找到指定时间的所有数据
            selected_data = news_df[news_df['datetime'] == target_time]

            selected_df = selected_data['title']

            # 将数据放入数组中
            selected_data_array = selected_df.values.tolist()

            aspect_token = self.tokenizer.tokenize(corporate_name)

            for select in selected_data_array:
                index = select.find(corporate_name)
                front_sentence = select[0:index]
                back_sentence = select[index+len(corporate_name):]
                front_token = self.tokenizer.tokenize(front_sentence)
                back_token =self.tokenizer.tokenize(back_sentence)
                asp_token = self.tokenizer.tokenize('<asp>')
                end_asp_token = self.tokenizer.tokenize('</asp>')
                token = front_token + asp_token + aspect_token + end_asp_token + back_token
                token = token[: 510]
                input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + token + ['[SEP]'])
                attention_mask = [1] * len(input_ids)
                input_ids = torch.tensor(input_ids)
                attention_mask = torch.tensor(attention_mask)
                # 计算需要填充的数量
                pad_length = padding_size - input_ids.size(0)

                # 对 input_ids 进行填充
                input_ids_padded = F.pad(input_ids, (0, pad_length), value=0)

                # 对 attention_mask 进行填充
                attention_mask_padded = F.pad(attention_mask, (0, pad_length), value=0)
                total_news_list.append(input_ids_padded)
                total_masks_list.append(attention_mask_padded)

                corporate_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + aspect_token + ['[SEP]'])
                corporate_masks = [1] * len(corporate_ids)

                corporate_ids = torch.tensor(corporate_ids)
                corporate_masks = torch.tensor(corporate_masks)
                #print(corporate_ids.shape)
                pad_length = padding_size - corporate_ids.size(0)

                corporate_ids_padded = F.pad(corporate_ids, (0, pad_length), value=0)
                corporate_masks_padded = F.pad(corporate_masks, (0, pad_length), value=0)

                total_corporates_list.append(corporate_ids_padded)
                total_corporate_masks_list.append(corporate_masks_padded)

                label_list.append(label_df)
            # print(len(return_list))

        return total_news_list, total_masks_list, total_corporates_list, total_corporate_masks_list, label_list

    def __getitem__(self, index):
        return self.new_data[index], self.mask_data[index], self.corporate_data[index], self.corporate_mask_data[index], self.label[index]

    def __len__(self):
        return self.len

class valDataset():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
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

        new_data = []
        mask_data = []
        corporate_data = []
        corporate_mask_data = []
        label = []

        for corporate in corporate_list:
            print(corporate)
            total_news_list, total_masks_list, total_corporates_list, total_corporate_masks_list, label_lists = self.get_Data(corporate)

            new_data = new_data + total_news_list
            mask_data = mask_data + total_masks_list
            corporate_data = corporate_data + total_corporates_list
            corporate_mask_data = corporate_mask_data + total_corporate_masks_list
            label = label + label_lists
        length = len(new_data)

        self.new_data = new_data
        self.mask_data = mask_data
        self.corporate_data = corporate_data
        self.corporate_mask_data = corporate_mask_data
        self.label = label
        self.len = length

    def get_Data(self, corporate_name):
        total_news_list = []
        total_masks_list = []
        total_corporates_list = []
        total_corporate_masks_list = []
        label_list = []

        # 定义填充大小
        padding_size = 512
        news_path = 'news_data/' + corporate_name + '.xlsx'
        news_df = pd.read_excel(news_path)
        news_df['datetime'] = pd.to_datetime(news_df['datetime']).dt.floor('d')
        policy_path = 'policy.xlsx'
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
        for i in range(0, len(data_df)):
            label = data_df.iloc[i, 58].item()
            if label >= 0.5:
                label = 1
            else:
                label = 0
            label_ls = [label]
            label_df = pd.DataFrame(label_ls, dtype=np.float32)
            label_df = torch.tensor(label_df.values)
            target_time = data_df.iloc[i, 0]
            # 找到指定时间的所有数据
            selected_data = news_df[news_df['datetime'] == target_time]

            selected_df = selected_data['title']

            # 将数据放入数组中
            selected_data_array = selected_df.values.tolist()

            aspect_token = self.tokenizer.tokenize(corporate_name)

            for select in selected_data_array:
                index = select.find(corporate_name)
                front_sentence = select[0:index]
                back_sentence = select[index+len(corporate_name):]
                front_token = self.tokenizer.tokenize(front_sentence)
                back_token =self.tokenizer.tokenize(back_sentence)
                asp_token = self.tokenizer.tokenize('<asp>')
                end_asp_token = self.tokenizer.tokenize('</asp>')
                token = front_token + asp_token + aspect_token + end_asp_token + back_token
                token = token[: 510]
                input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + token + ['[SEP]'])
                attention_mask = [1] * len(input_ids)
                input_ids = torch.tensor(input_ids)
                attention_mask = torch.tensor(attention_mask)
                # 计算需要填充的数量
                pad_length = padding_size - input_ids.size(0)

                # 对 input_ids 进行填充
                input_ids_padded = F.pad(input_ids, (0, pad_length), value=0)

                # 对 attention_mask 进行填充
                attention_mask_padded = F.pad(attention_mask, (0, pad_length), value=0)
                total_news_list.append(input_ids_padded)
                total_masks_list.append(attention_mask_padded)

                corporate_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + aspect_token + ['[SEP]'])
                corporate_masks = [1] * len(corporate_ids)

                corporate_ids = torch.tensor(corporate_ids)
                corporate_masks = torch.tensor(corporate_masks)
                #print(corporate_ids.shape)
                pad_length = padding_size - corporate_ids.size(0)

                corporate_ids_padded = F.pad(corporate_ids, (0, pad_length), value=0)
                corporate_masks_padded = F.pad(corporate_masks, (0, pad_length), value=0)

                total_corporates_list.append(corporate_ids_padded)
                total_corporate_masks_list.append(corporate_masks_padded)

                label_list.append(label_df)
            # print(len(return_list))

        return total_news_list, total_masks_list, total_corporates_list, total_corporate_masks_list, label_list

    def __getitem__(self, index):
        return self.new_data[index], self.mask_data[index], self.corporate_data[index], self.corporate_mask_data[index], self.label[index]

    def __len__(self):
        return self.len

class testDataset():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
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
        graph_data = []
        new_data = []
        mask_data = []
        corporate_data = []
        corporate_mask_data = []
        policies_data = []
        policy_masks_data = []
        label = []

        for corporate in corporate_list:
            print(corporate)
            return_list, graph_list, total_news_list, total_masks_list, total_corporates_list, total_corporate_masks_list, total_policies_list, \
                total_policy_masks_list, label_lists = self.get_Data(corporate)
            data_list, the_graph_list ,new_list, mask_list, corporates_list, corporate_masks_list, policies_list, policy_masks_list, label_list = \
                self.slide_windows(return_list, graph_list, total_news_list, total_masks_list, total_corporates_list,total_corporate_masks_list,
                                   total_policies_list, total_policy_masks_list, label_lists)
            data = data + data_list
            graph_data = graph_data + the_graph_list
            new_data = new_data + new_list
            mask_data = mask_data + mask_list
            corporate_data = corporate_data + corporates_list
            corporate_mask_data = corporate_mask_data + corporate_masks_list
            policies_data = policies_data + policies_list
            policy_masks_data = policy_masks_data + policy_masks_list
            label = label + label_list
        length = len(data)

        self.data = data
        self.graph_data = graph_data
        self.new_data = new_data
        self.mask_data = mask_data
        self.corporate_data = corporate_data
        self.corporate_mask_data = corporate_mask_data
        self.policies_data = policies_data
        self.policy_masks_data = policy_masks_data
        self.label = label
        self.len = length


    # def create_node_masked_graph(self, graph_data):
    #     masked_data = HeteroData()
    #
    #     # 复制节点特征
    #     for key, value in graph_data.x_dict.items():
    #         masked_data[key].x = value
    #
    #     # 随机 mask 掉节点
    #     num_nodes = masked_data.num_nodes
    #     mask_nodes = random.sample(range(num_nodes), num_nodes // 2)  # 随机选择一半节点进行 mask
    #     for i in range(num_nodes):
    #         for j in range(num_nodes):
    #             src_type = f'node_type_{i}'
    #             dst_type = f'node_type_{j}'
    #             if i not in mask_nodes and j not in mask_nodes and i != j:  # 排除被 mask 的节点和自环边
    #                 # 复制边到被 mask 的图中
    #                 masked_data.edge(src_type, dst_type, torch.tensor([i]), torch.tensor([j]))
    #
    #     return masked_data
    #
    # # 创建随机 mask 掉边的图
    # def create_edge_masked_graph(self, graph_data):
    #     masked_data = HeteroData()
    #
    #     # 复制节点特征
    #     for key, value in graph_data.x_dict.items():
    #         masked_data[key].x = value
    #
    #     # 复制节点边的信息，并随机 mask 掉一半的边
    #     for key, value in graph_data.edge_index_dict.items():
    #         src_nodes, dst_nodes = value[0], value[1]
    #         num_edges = src_nodes.size(0)
    #         mask_edges = random.sample(range(num_edges), num_edges // 2)  # 随机选择一半边进行 mask
    #         masked_src_nodes = [src_nodes[i] for i in range(num_edges) if i not in mask_edges]
    #         masked_dst_nodes = [dst_nodes[i] for i in range(num_edges) if i not in mask_edges]
    #         masked_data.edge_index(key[0], key[1], torch.tensor(masked_src_nodes), torch.tensor(masked_dst_nodes))
    #
    #     return masked_data

    def get_Data(self, corporate_name):
        return_list = []
        graph_list = []
        total_news_list = []
        total_masks_list = []
        total_corporates_list = []
        total_corporate_masks_list = []
        total_policies_list = []
        total_policy_masks_list = []
        label_list = []

        # 定义填充大小
        padding_size = 512
        padding_len = 10
        news_path = 'news_data/' + corporate_name + '.xlsx'
        news_df = pd.read_excel(news_path)
        news_df['datetime'] = pd.to_datetime(news_df['datetime']).dt.floor('d')
        policy_path = 'policy.xlsx'
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
        for i in range(0, len(data_df)):
            # print(i)
            '''
            先测试小数据，原数据为486
            '''
            # current_df = data_df.iloc[i, :]
            # print(current_df)
            #print(i)
            stock_df = pd.concat([data_df.iloc[i, 1:6], data_df.iloc[i, 43:48]], axis=0)
            financial_df = data_df.iloc[i, 6:22]
            macro_df = data_df.iloc[i, 22:43]
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
            stock_df = stock_df.view(1,-1)
            financial_df = financial_df.view(1,-1)
            macro_df = macro_df.view(1,-1)
            label_df = label_df.view(1,-1)
            #print(stock_df.shape)
            target_time = data_df.iloc[i, 0]
            # 找到指定时间的所有数据
            selected_data = news_df[news_df['datetime'] == target_time]

            selected_df = selected_data['title']

            # 将数据放入数组中
            selected_data_array = selected_df.values.tolist()

            aspect_token = self.tokenizer.tokenize(corporate_name)

            news_list = []
            masks_list = []
            corporates_list = []
            corporate_masks_list = []

            for select in selected_data_array:
                index = select.find(corporate_name)
                front_sentence = select[0:index]
                back_sentence = select[index+len(corporate_name):]
                front_token = self.tokenizer.tokenize(front_sentence)
                back_token =self.tokenizer.tokenize(back_sentence)
                asp_token = self.tokenizer.tokenize('<asp>')
                end_asp_token = self.tokenizer.tokenize('</asp>')
                token = front_token + asp_token + aspect_token + end_asp_token + back_token
                token = token[: 510]
                input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + token + ['[SEP]'])
                attention_mask = [1] * len(input_ids)
                input_ids = torch.tensor(input_ids)
                attention_mask = torch.tensor(attention_mask)
                # 计算需要填充的数量
                pad_length = padding_size - input_ids.size(0)

                # 对 input_ids 进行填充
                input_ids_padded = F.pad(input_ids, (0, pad_length), value=0)

                # 对 attention_mask 进行填充
                attention_mask_padded = F.pad(attention_mask, (0, pad_length), value=0)
                news_list.append(input_ids_padded)
                masks_list.append(attention_mask_padded)

                corporate_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + aspect_token + ['[SEP]'])
                corporate_masks = [1] * len(corporate_ids)

                corporate_ids = torch.tensor(corporate_ids)
                corporate_masks = torch.tensor(corporate_masks)
                #print(corporate_ids.shape)
                pad_length = padding_size - corporate_ids.size(0)

                corporate_ids_padded = F.pad(corporate_ids, (0, pad_length), value=0)
                corporate_masks_padded = F.pad(corporate_masks, (0, pad_length), value=0)

                corporates_list.append(corporate_ids_padded)
                corporate_masks_list.append(corporate_masks_padded)

            target_time = pd.to_datetime(target_time).to_period('M')

            selected_policy = policy_df[policy_df['时间'] == target_time]
            #print(selected_policy)

            policy_data = selected_policy['政策'].iloc[0]
            #print(policy_data)
            policy_token = self.tokenizer.tokenize(policy_data)
            policy_token = policy_token[: 510]
            policy_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + policy_token + ['[SEP]'])
            policy_masks = [1] * len(policy_ids)
            policy_ids = torch.tensor(policy_ids)
            policy_masks = torch.tensor(policy_masks)
            pad_length = padding_size - policy_ids.size(0)
            # 对 input_ids 进行填充
            policy_ids_padded = F.pad(policy_ids, (0, pad_length), value=0)
            # 对 attention_mask 进行填充
            policy_mask_padded = F.pad(policy_masks, (0, pad_length), value=0)

            tool_ones = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
            tool_zeros = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            record_policy = 1
            record_news = 1

            if policy_data == '无':
                record_policy = 0
            if len(news_list) == 0:
                record_news = 0

            record = 10 * record_news + record_policy

            # 创建异构图对象
            data_graph = Data()
            cnt_tensor = torch.rand(1, 3)
            '''
            要更正
            '''

            if record == 11:
                graph_tensor = torch.cat((tool_ones, tool_ones, tool_ones, tool_ones, tool_ones), dim=1)

                # 创建新的数据对象，对节点和边进行掩码
                masked_graph_tensor_1 = graph_tensor.clone().view(5, 10)  # 复制数据对象
                masked_graph_tensor_2 = graph_tensor.clone().view(5, 10)  # 复制数据对象

                # 随机选择两行索引
                rows_to_mask = torch.randperm(5)[:2]

                # 将选择的行置为 0
                masked_graph_tensor_1[rows_to_mask] = 0

                # 随机选择两行索引
                rows_to_mask = torch.randperm(5)[:2]

                # 将选择的行置为 0
                masked_graph_tensor_2[rows_to_mask] = 0

                masked_graph_tensor_1 = masked_graph_tensor_1.view(1, -1)
                masked_graph_tensor_2 = masked_graph_tensor_2.view(1, -1)

                # # 对节点进行掩码操作
                # for key in masked_node_data.keys():
                #     if key.endswith('_x'):
                #         mask = torch.rand_like(masked_node_data[key]) > 0.5  # 生成与节点特征相同形状的随机掩码
                #         masked_node_data[key] = masked_node_data[key] * mask  # 对节点特征进行掩码操作
                #
                # # 对边进行掩码操作
                # for key in masked_edge_data.edge_index_dict.keys():
                #     edge_index = masked_edge_data.edge_index_dict[key]
                #     mask = torch.rand_like(edge_index.to(torch.float)) > 0.5  # 生成与边索引相同形状的随机掩码
                #     masked_edge_data.edge_index_dict[key] = edge_index * mask  # 对边索引进行掩码

            elif record == 10:
                graph_tensor = torch.cat((tool_ones, tool_ones, tool_ones, tool_ones), dim=1)

                # 创建新的数据对象，对节点和边进行掩码
                masked_graph_tensor_1 = graph_tensor.clone().view(4, 10)  # 复制数据对象
                masked_graph_tensor_2 = graph_tensor.clone().view(4, 10)  # 复制数据对象

                # 随机选择两行索引
                rows_to_mask = torch.randperm(4)[:2]

                # 将选择的行置为 0
                masked_graph_tensor_1[rows_to_mask] = 0

                # 随机选择两行索引
                rows_to_mask = torch.randperm(4)[:2]

                # 将选择的行置为 0
                masked_graph_tensor_2[rows_to_mask] = 0

                graph_tensor = torch.cat((graph_tensor, tool_zeros), dim=1)
                masked_graph_tensor_1 = masked_graph_tensor_1.view(1, -1)
                masked_graph_tensor_2 = masked_graph_tensor_2.view(1, -1)

                masked_graph_tensor_1 = torch.cat((masked_graph_tensor_1, tool_zeros), dim=1)
                masked_graph_tensor_2 = torch.cat((masked_graph_tensor_2, tool_zeros), dim=1)
            elif record == 1:
                graph_tensor = torch.cat((tool_ones, tool_ones, tool_ones, tool_ones), dim=1)

                # 创建新的数据对象，对节点和边进行掩码
                masked_graph_tensor_1 = graph_tensor.clone().view(4, 10)  # 复制数据对象
                masked_graph_tensor_2 = graph_tensor.clone().view(4, 10)  # 复制数据对象

                # 随机选择两行索引
                rows_to_mask = torch.randperm(4)[:2]

                # 将选择的行置为 0
                masked_graph_tensor_1[rows_to_mask] = 0

                # 随机选择两行索引
                rows_to_mask = torch.randperm(4)[:2]

                # 将选择的行置为 0
                masked_graph_tensor_2[rows_to_mask] = 0

                tensor_part1_3 = masked_graph_tensor_1[:3]  # 提取第1到第3行
                tensor_part4 = masked_graph_tensor_1[3:]  # 提取第4行

                # 执行拼接操作
                masked_graph_tensor_1 = torch.cat((tensor_part1_3, tool_zeros, tensor_part4), dim=0)

                tensor_part1_3 = masked_graph_tensor_2[:3]  # 提取第1到第3行
                tensor_part4 = masked_graph_tensor_2[3:]  # 提取第4行

                # 执行拼接操作
                masked_graph_tensor_2 = torch.cat((tensor_part1_3, tool_zeros, tensor_part4), dim=0)

                graph_tensor = torch.cat((tool_ones, tool_ones, tool_ones, tool_zeros, tool_ones), dim=1)
                masked_graph_tensor_1 = masked_graph_tensor_1.view(1, -1)
                masked_graph_tensor_2 = masked_graph_tensor_2.view(1, -1)
            else:
                graph_tensor = torch.cat((tool_ones, tool_ones, tool_ones), dim=1)

                # 创建新的数据对象，对节点和边进行掩码
                masked_graph_tensor_1 = graph_tensor.clone().view(3, 10)  # 复制数据对象
                masked_graph_tensor_2 = graph_tensor.clone().view(3, 10)  # 复制数据对象

                # 随机选择两行索引
                rows_to_mask = torch.randperm(3)[:1]

                # 将选择的行置为 0
                masked_graph_tensor_1[rows_to_mask] = 0

                # 随机选择两行索引
                rows_to_mask = torch.randperm(3)[:1]

                # 将选择的行置为 0
                masked_graph_tensor_2[rows_to_mask] = 0

                masked_graph_tensor_1 = masked_graph_tensor_1.view(1, -1)
                masked_graph_tensor_2 = masked_graph_tensor_2.view(1, -1)

                graph_tensor = torch.cat((graph_tensor, tool_zeros, tool_zeros), dim=1)
                masked_graph_tensor_1 = torch.cat((masked_graph_tensor_1, tool_zeros, tool_zeros), dim=1)
                masked_graph_tensor_2 = torch.cat((masked_graph_tensor_2, tool_zeros, tool_zeros), dim=1)

            #print(news_list)

            # 如果 news_list 为空，则构建一个 1*512 的全0张量
            if not news_list:
                news_tensor = torch.zeros(1, 512)
            if news_list:
                news_tensor = torch.stack(news_list)
            if not masks_list:
                masks_tensor = torch.zeros(1, 512)
            if masks_list:
                masks_tensor = torch.stack(masks_list)
            if not corporates_list:
                corporates_tensor = torch.zeros(1, 512)
            if corporates_list:
                corporates_tensor = torch.stack(corporates_list)
            if not corporate_masks_list:
                corporate_masks_tensor = torch.zeros(1, 512)
            if corporate_masks_list:
                corporate_masks_tensor = torch.stack(corporate_masks_list)
            current_len = len(news_list)
            if not news_list:
                current_len = 1
            pad_size = padding_len - current_len
            if pad_size >= 0:
                # 检查张量的维度
                # if news_tensor.dim() == 1:
                #     news_tensor = news_tensor.unsqueeze(0)  # 在第一个维度上添加一个维度
                #print(news_tensor.shape)
                # 对 news_tensor 进行填充
                news_tensor_padded = F.pad(news_tensor, (0, 0, 0, pad_size), value=0)
                # 对 masks_tensor 进行填充
                masks_tensor_padded = F.pad(masks_tensor, (0, 0, 0, pad_size), value=0)
                #print(news_tensor_padded.shape)
                corporates_tensor_padded = F.pad(corporates_tensor, (0, 0, 0, pad_size), value=0)

                corporate_masks_tensor_padded = F.pad(corporate_masks_tensor, (0, 0, 0, pad_size), value=0)

            else:
                news_tensor_padded = news_tensor[:10, :]
                masks_tensor_padded = masks_tensor[:10, :]
                corporates_tensor_padded = corporates_tensor[:10, :]

                corporate_masks_tensor_padded = corporate_masks_tensor[:10, :]

            stock_df = stock_df.view(1, -1)
            financial_df = financial_df.view(1, -1)
            macro_df = macro_df.view(1, -1)
            # print("stock:", stock_df.shape)
            # print("financial:", financial_df.shape)
            # print("macro:", macro_df.shape)
            total_df = torch.cat((stock_df,financial_df,macro_df), dim=1)

            return_list.append(total_df)
            # print(graph_tensor.shape)
            # print(masked_graph_tensor_1.shape)
            graph_list.append(torch.cat((graph_tensor, masked_graph_tensor_1, masked_graph_tensor_2), dim=1))
            total_news_list.append(news_tensor_padded)
            total_masks_list.append(masks_tensor_padded)
            total_corporates_list.append(corporates_tensor_padded)
            total_corporate_masks_list.append(corporate_masks_tensor_padded)
            total_policies_list.append(policy_ids_padded)
            total_policy_masks_list.append(policy_mask_padded)
            label_list.append(label_df)
            # print(len(return_list))

        return return_list, graph_list, total_news_list, total_masks_list, total_corporates_list, total_corporate_masks_list, total_policies_list, total_policy_masks_list, label_list

    def slide_windows(self, return_list, graph_list, total_news_list, total_masks_list, total_corporates_list, total_corporate_masks_list, total_policies_list, total_policy_masks_list, label_lists):
        data = []
        graph_data = []
        new_data = []
        mask_data = []
        corporate_data = []
        corporate_masks_data = []
        policies_data = []
        policy_masks_data = []
        label = []
        for i in range(0, len(return_list) - self.stride, self.stride):
            #print(return_list[i:i + self.stride])
            return_df = pd.DataFrame([tensor.flatten().numpy() for tensor in return_list[i:i + self.stride]], dtype=np.float32)
            return_df = torch.tensor(return_df.values)
            data.append(return_df)

            total_graph_df = pd.DataFrame([tensor.flatten().numpy() for tensor in graph_list[i:i + self.stride]], dtype=np.int)
            total_graph_df = torch.tensor(total_graph_df.values)
            graph_data.append(total_graph_df)

            total_news_df = pd.DataFrame([tensor.flatten().numpy() for tensor in total_news_list[i:i + self.stride]], dtype=np.int)
            total_news_df = torch.tensor(total_news_df.values)
            new_data.append(total_news_df)
            total_masks_df = pd.DataFrame([tensor.flatten().numpy() for tensor in total_masks_list[i:i + self.stride]], dtype=np.int)
            total_masks_df = torch.tensor(total_masks_df.values)
            mask_data.append(total_masks_df)
            total_corporates_df = pd.DataFrame([tensor.flatten().numpy() for tensor in total_corporates_list[i:i + self.stride]], dtype=np.int)
            total_corporates_df = torch.tensor(total_corporates_df.values)
            corporate_data.append(total_corporates_df)
            total_corporate_masks_df = pd.DataFrame([tensor.flatten().numpy() for tensor in total_corporate_masks_list[i:i + self.stride]], dtype=np.int)
            total_corporate_masks_df = torch.tensor(total_corporate_masks_df.values)
            corporate_masks_data.append(total_corporate_masks_df)
            total_policies_df = pd.DataFrame([tensor.flatten().numpy() for tensor in total_policies_list[i:i + self.stride]], dtype=np.int)
            total_policies_df = torch.tensor(total_policies_df.values)
            policies_data.append(total_policies_df)
            total_policy_masks_df = pd.DataFrame([tensor.flatten().numpy() for tensor in total_policy_masks_list[i:i + self.stride]], dtype=np.int)
            total_policy_masks_df = torch.tensor(total_policy_masks_df.values)
            policy_masks_data.append(total_policy_masks_df)
            label_df = pd.DataFrame([tensor.flatten().numpy() for tensor in label_lists[i + self.stride]], dtype=np.float32)
            label_df = torch.tensor(label_df.values)
            label.append(label_df)

        return data, graph_data, new_data, mask_data, corporate_data, corporate_masks_data, policies_data, policy_masks_data,label

    def __getitem__(self, index):
        return self.data[index], self.graph_data[index], self.new_data[index], self.mask_data[index], self.corporate_data[index], self.corporate_mask_data[index], \
            self.policies_data[index], self.policy_masks_data[index],self.label[index]

    def __len__(self):
        return self.len