import torch
import pandas as pd
import sklearn
from transformers import BertModel,BertTokenizer
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('./bert-base-Chinese')
model = BertModel.from_pretrained('./bert-base-Chinese')
model.eval()

def encode_text(text, corporate_name, model, tokenizer):
    index = text.find(corporate_name)
    front_sentence = text[0:index]
    back_sentence = text[index + len(corporate_name):]
    front_token = tokenizer.tokenize(front_sentence)
    back_token = tokenizer.tokenize(back_sentence)
    asp_token = tokenizer.tokenize('<asp>')
    end_asp_token = tokenizer.tokenize('</asp>')
    aspect_token = tokenizer.tokenize(corporate_name)
    token = front_token + asp_token + aspect_token + end_asp_token + back_token
    token = token[: 510]
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + token + ['[SEP]'])
    attention_mask = [1] * len(input_ids)
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    input_ids = input_ids.view(1, -1)
    attention_mask = attention_mask.view(1, -1)
    outputs = model(input_ids, attention_mask=attention_mask)
    # print(outputs.last_hidden_state)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()


# for corporate_name in corporate_list:
#     news_path = 'news_data/' + corporate_name + '.xlsx'
#     news_df = pd.read_excel(news_path)
#     news_df['datetime'] = pd.to_datetime(news_df['datetime']).dt.floor('d')
#     encoded_news = []
#     for news_text in news_df['title']:
#         encoded_news_text = encode_text(news_text, corporate_name, tokenizer=tokenizer, model=model)
#         # print(encoded_news_text)
#         encoded_news.append(encoded_news_text)
#
#     # 将列表转换为 NumPy 数组
#     encoded_news_array = np.array(encoded_news)
#     # print(encoded_news_array)
#     # 获取唯一的日期列表
#     unique_dates = news_df['datetime'].unique()
#
#     # 创建一个新的列表，存储每天的新闻表示的平均值
#     average_news_representations = []
#     for date in unique_dates:
#         # 获取特定日期的索引
#         indices = news_df[news_df['datetime'] == date].index
#
#         # 获取特定日期的新闻表示
#         news_representations = encoded_news_array[indices]
#
#         # 计算特定日期的新闻表示的平均值，并存储结果
#         average_news_representations.append(news_representations.mean(axis=0))
#
#     # 将列表转换为 NumPy 数组
#     average_news_representations_array = np.array(average_news_representations)
#     average_news_representations_data = pd.DataFrame(average_news_representations_array)
#
#     average_news_representations_data['date'] = unique_dates
#     # 存储结果
#     average_news_representations_data.to_excel('corporates_news/' + corporate_name + '.xlsx', index=False)
#
#     print(corporate_name)


    # 创建一个新的列表，存储每天的新闻表示的平均值
average_news_representations = []
policy_path = 'policy.xlsx'
policy_df = pd.read_excel(policy_path)
for policy_text in policy_df['政策']:
    policy_token = tokenizer.tokenize(policy_text)
    policy_token = policy_token[: 510]
    policy_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + policy_token + ['[SEP]'])
    policy_masks = [1] * len(policy_ids)
    policy_ids = torch.tensor(policy_ids)
    policy_ids = policy_ids.view(1, -1)
    policy_masks = torch.tensor(policy_masks)
    policy_masks = policy_masks.view(1, -1)
    outputs = model(policy_ids, attention_mask=policy_masks)
    # print(outputs.last_hidden_state)
    output = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()


    average_news_representations.append(output)

# 将列表转换为 NumPy 数组
average_news_representations_array = np.array(average_news_representations)
average_news_representations_data = pd.DataFrame(average_news_representations_array)

average_news_representations_data.to_excel('policy_tensor.xlsx', index=False)

