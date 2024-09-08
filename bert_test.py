import bert_data
import bert_model
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertModel,BertTokenizer
import torch.nn.functional as F
import os
from torch_geometric.data import DataLoader

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

def attention(query, key):
    # 计算query和key之间的点积
    attn_weights = torch.matmul(query, key.transpose(-2, -1))
    # 使用softmax函数对注意力权重进行归一化
    attn_weights = F.softmax(attn_weights, dim=-1)
    # 对key进行加权求和，得到注意力池化的结果
    attn_output = torch.matmul(attn_weights, key)
    return attn_output

def get_Data(corporate_name, mymodel):
    df = pd.DataFrame(columns=['Time', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
    # 定义填充大小
    padding_size = 512
    news_path = 'news_data/' + corporate_name + '.xlsx'
    news_df = pd.read_excel(news_path)
    news_df['datetime'] = pd.to_datetime(news_df['datetime']).dt.floor('d')

    file_path = 'stock_data/2022-4/测试集/A股/' + corporate_name + '.xlsx'
    if os.path.exists(file_path):
        data_df = pd.read_excel(file_path)
    else:
        file_path = 'stock_data/2022-4/测试集/港股/' + corporate_name + '.xlsx'
        data_df = pd.read_excel(file_path)
    data_df['date'] = pd.to_datetime(data_df['date']).dt.floor('d')
    for i in range(0, len(data_df)):
        target_time = data_df.iloc[i, 0]
        # 找到指定时间的所有数据
        selected_data = news_df[news_df['datetime'] == target_time]

        selected_df = selected_data['title']

        # 将数据放入数组中
        selected_data_array = selected_df.values.tolist()

        aspect_token = tokenizer.tokenize(corporate_name)

        cnt_tensor = torch.zeros(1, 10)
        cnt_list = []

        if len(selected_data_array) == 0:
            continue

        for select in selected_data_array:
            index = select.find(corporate_name)
            front_sentence = select[0:index]
            back_sentence = select[index + len(corporate_name):]
            front_token = tokenizer.tokenize(front_sentence)
            back_token = tokenizer.tokenize(back_sentence)
            asp_token = tokenizer.tokenize('<asp>')
            end_asp_token = tokenizer.tokenize('</asp>')
            token = front_token + asp_token + aspect_token + end_asp_token + back_token
            token = token[: 510]
            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + token + ['[SEP]'])
            attention_mask = [1] * len(input_ids)
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            # 计算需要填充的数量
            pad_length = padding_size - input_ids.size(0)

            # 对 input_ids 进行填充
            input_ids_padded = F.pad(input_ids, (0, pad_length), value=0)

            # 对 attention_mask 进行填充
            attention_mask_padded = F.pad(attention_mask, (0, pad_length), value=0)

            corporate_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + aspect_token + ['[SEP]'])
            corporate_masks = [1] * len(corporate_ids)

            corporate_ids = torch.tensor(corporate_ids)
            corporate_masks = torch.tensor(corporate_masks)
            # print(corporate_ids.shape)
            pad_length = padding_size - corporate_ids.size(0)

            corporate_ids_padded = F.pad(corporate_ids, (0, pad_length), value=0)
            corporate_masks_padded = F.pad(corporate_masks, (0, pad_length), value=0)

            input_ids_padded = input_ids_padded.view(1,-1).to('cuda:1')
            attention_mask_padded = attention_mask_padded.view(1,-1).to('cuda:1')
            corporate_ids_padded = corporate_ids_padded.view(1, -1).to('cuda:1')
            corporate_masks_padded = corporate_masks_padded.view(1, -1).to('cuda:1')

            output, output_tensor = mymodel(input_ids_padded, attention_mask_padded, corporate_ids_padded, corporate_masks_padded)

            cnt_tensor = cnt_tensor.to('cuda:1')
            cnt_tensor = cnt_tensor + output_tensor
            cnt_list.append(output_tensor)

        cnt_tensor_total = cnt_tensor / len(selected_data_array)
        # 初始化一个列表，用来存放每个tensor与tensor0的注意力池化结果
        attention_outputs = []

        # 遍历数组中的每个tensor
        for tensor in cnt_list:
            # 计算注意力池化结果，并添加到列表中
            attn_output = attention(cnt_tensor_total, tensor)
            attention_outputs.append(attn_output)

        # 将列表转换为tensor，并沿着0维度求和，得到最终的全局注意力池化结果
        global_attention = torch.stack(attention_outputs).sum(dim=0)

        total_tensor = torch.cat((cnt_tensor_total, global_attention), dim=1)
        tensor_values = total_tensor.cpu().detach().numpy().flatten().tolist()
        row_data = [target_time] + tensor_values
        df.loc[len(df)] = row_data
    df.to_excel('corporates_daily/' + corporate_name + '.xlsx',index = False)
    return


device = 'cuda:1'
# 添加不同类型的节点
bert = BertModel.from_pretrained('./bert-base-Chinese')
# 参数都得改
mymodel = bert_model.BertGNNGru(bert_with_layer_size=10, news_size=10, bert=bert, num_heads=1).to(device)
mymodel.load_state_dict(torch.load('model_bert.pth'))

for corporate in corporate_list_0:
    get_Data(corporate_name=corporate, mymodel=mymodel)