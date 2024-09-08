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
device = 'cuda'
for corporate in corporate_list:
    mydataset = dataseter.valDataset(corporate=corporate)
    # 参数都得改
    mymodel = model.testmodel(57).to(device)
    mymodel.load_state_dict(torch.load('model_state_dict_gru.pth'))
    transmodel = model.StockTransformer(57, 1).to(device)
    transmodel.load_state_dict(torch.load('model_state_dict_trans.pth'))
    # Define LSTM model
    lstm_model = model.StockLSTM(57, 64, 1, 2).to(device)
    lstm_model.load_state_dict(torch.load('model_state_dict_lstm.pth'))
    # Define GRU model
    gru_model = model.StockGRU(57, 64, 1, 2).to(device)
    gru_model.load_state_dict(torch.load('model_state_dict_gru_tradition.pth'))
    outputs_mymodel = []
    outputs_trans = []
    outputs_gru = []
    outputs_lstm = []
    labels = []
    mymodel.eval()
    transmodel.eval()
    gru_model.eval()
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
        output_mymodel = mymodel(data, news)
        output_trans = transmodel(data)
        output_gru = gru_model(data)
        output_lstm = lstm_model(data)
        output_mymodel = output_mymodel.item()
        output_trans = output_trans.item()
        output_gru = output_gru.item()
        output_lstm = output_lstm.item()

        label = label.item()
        outputs_mymodel.append(output_mymodel)
        outputs_trans.append(output_trans)
        outputs_gru.append(output_gru)
        outputs_lstm.append(output_lstm)
        labels.append(label)

    x_values = range(1, len(mydataset) + 1)

    outputs_mymodel = np.array(outputs_mymodel).reshape(1, -1)
    outputs_trans = np.array(outputs_trans).reshape(1, -1)
    outputs_gru = np.array(outputs_gru).reshape(1, -1)
    outputs_lstm = np.array(outputs_lstm).reshape(1, -1)
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

    outputs_mymodel = outputs_mymodel.flatten().tolist()
    outputs_trans = outputs_trans.flatten().tolist()
    outputs_gru = outputs_gru.flatten().tolist()
    outputs_lstm = outputs_lstm.flatten().tolist()
    labels = labels.flatten().tolist()

    # outputs = np.array(outputs).reshape(1, -1)
    # outputs = y_scaler.inverse_transform(outputs)
    # outputs = outputs.flatten().tolist()
    # labels = np.array(labels).reshape(1, -1)
    # labels = y_scaler.inverse_transform(labels)
    # labels = labels.flatten().tolist()
    plt.figure(figsize=(12, 8))
    plt.plot(x_values, labels, color='black', label='Label Value')  # 第一个y值用红色表示
    plt.plot(x_values, outputs_mymodel, color='red', label=' Outputs Value')  # 第二个y值用绿色表示
    plt.plot(x_values, outputs_trans, color='blue', label=' Trans Value')  # 第二个y值用绿色表示
    plt.plot(x_values, outputs_gru, color='green', label=' Grus Value')  # 第二个y值用绿色表示
    plt.plot(x_values, outputs_lstm, color='yellow', label=' Lstms Value')  # 第二个y值用绿色表示


    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title('Y Values Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Y Values')

    # 保存为图片
    plt.savefig('corporate_figures/' + corporate + '.png')