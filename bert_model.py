import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertLayer
from torch_geometric.nn import GATConv,HANConv
import math
from torch.autograd import Variable
import pandas as pd
import numpy as np

'''
可能实际使用中没有batch_size，后面要加上
'''
class BertWithLayer(nn.Module):
    def __init__(self, output_size, bert, num_layers=1):
        super(BertWithLayer, self).__init__()
        self.bert = bert
        for name, para in self.bert.named_parameters():
            para.requires_grad_(False)
        self.layer = BertLayer(self.bert.config)
        for name, para in self.layer.named_parameters():
            para.requires_grad_(False)
        self.fc = nn.Linear(768,output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        processed_hidden_state = last_hidden_state.clone()

        #print(last_hidden_state.shape)
        for i in range(last_hidden_state.size(1)):
            if i == 0:
                    #print(last_hidden_state[:, i, :].unsqueeze(1).shape)
                    #layer(last_hidden_state[:, i, :], attention_mask[:, i, :])
                a = processed_hidden_state[:, i, :].unsqueeze(1)
                    #b = attention_mask[:, i, :]
                    #print(layer(a))
                last_hidden_state[:, i, :] = self.layer(a)[0].squeeze(1)

                dense_tensor = last_hidden_state[:, i, :].to('cuda:1')
                # 计算每个向量的各位加和
                sum_elements = torch.sum(dense_tensor, dim=1)

                # 找到非全零的向量
                non_zero_indices = torch.nonzero(sum_elements).squeeze()

                process_tensor = torch.randn(dense_tensor.shape[0], 10)

                # 对非全零的向量进行 softmax
                softmaxed_tensor_array = torch.zeros_like(process_tensor).to('cuda:1')
                if non_zero_indices.numel() > 0:
                    softmaxed_tensor_array[non_zero_indices] = nn.functional.softmax(self.fc(dense_tensor[non_zero_indices]), dim=-1)

                # 最后将全零向量位置保持为零
                softmaxed_tensor_array[sum_elements == 0] = 0
                dense_tensor_soft = softmaxed_tensor_array
            else:
                last_hidden_state[:, i, :] = self.layer(last_hidden_state[:, i, :].unsqueeze(1) + last_hidden_state[:, i - 1, :].unsqueeze(1))[0].squeeze(1)
                dense_tensor = last_hidden_state[:, i, :].to('cuda:1')
                # 计算每个向量的各位加和
                sum_elements = torch.sum(dense_tensor, dim=1)

                # 找到非全零的向量
                non_zero_indices = torch.nonzero(sum_elements).squeeze()
                process_tensor = torch.randn(dense_tensor.shape[0], 10)

                # 对非全零的向量进行 softmax
                softmaxed_tensor_array = torch.zeros_like(process_tensor).to('cuda:1')
                if non_zero_indices.numel() > 0:
                    softmaxed_tensor_array[non_zero_indices] = nn.functional.softmax(self.fc(dense_tensor[non_zero_indices]), dim=-1)

                # 最后将全零向量位置保持为零
                softmaxed_tensor_array[sum_elements == 0] = 0
                dense_tensor_soft = dense_tensor_soft + softmaxed_tensor_array
        # 计算每个向量的各位加和
        sum_of_elements = torch.sum(dense_tensor_soft, dim=1)

        # 对每个 b 向量进行归一化
        normalized_tensor_array = dense_tensor_soft / sum_of_elements.view(-1, 1)

        # 找到input_ids和attention_mask全为0的行
        all_zeros = (input_ids == 0) & (attention_mask == 0)
        #print(all_zeros.shape)
        #print(normalized_tensor_array.shape)
        # 调整 all_zeros 的形状以匹配 normalized_tensor_array
        all_zeros_plus = torch.all(all_zeros == 0, dim=1)
        output_matrix = torch.zeros_like(normalized_tensor_array[:, :10])  # 创建全0矩阵，形状为 (90, 10)
        output_matrix[all_zeros_plus, :] = 1  # 将全0行的对应行设为1
        #all_zeros = all_zeros.unsqueeze(-1).repeat(1, 1, normalized_tensor_array.size(-1))
        #print(all_zeros.shape)
        # 使用 masked_fill 方法将满足条件的行的张量置为0
        normalized_tensor_array = normalized_tensor_array.masked_fill(output_matrix.bool(), 0)
        # 将满足条件的行的encoded_tensors置为0
        # normalized_tensor_array[all_zeros.unsqueeze(-1).repeat(1, 1, normalized_tensor_array.size(-1))] = 0
        return normalized_tensor_array

class BertWithoutLayer(nn.Module):
    def __init__(self, output_size, bert):
        super(BertWithoutLayer, self).__init__()
        self.bert = bert
        for name, para in self.bert.named_parameters():
            para.requires_grad_(False)
        self.fc = nn.Linear(768,output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        for i in range(last_hidden_state.size(1)):
            dense_tensor = last_hidden_state[:, i, :]
            # 计算每个向量的各位加和
            sum_elements = torch.sum(dense_tensor, dim=1)

            # 找到非全零的向量
            non_zero_indices = torch.nonzero(sum_elements).squeeze()

            process_tensor = torch.randn(dense_tensor.shape[0], 10)

            # 对非全零的向量进行 softmax
            tensor_array = torch.zeros_like(process_tensor).to('cuda:1')
            if non_zero_indices.numel() > 0:
                tensor_array[non_zero_indices] = self.fc(dense_tensor[non_zero_indices])

            # 最后将全零向量位置保持为零
            tensor_array[sum_elements == 0] = 0
            dense_tensor_soft = tensor_array
        # 计算每个向量的各位加和
        sum_of_elements = torch.sum(dense_tensor_soft, dim=1)

        # 对每个 b 向量进行归一化
        normalized_tensor_array = dense_tensor_soft / sum_of_elements.view(-1, 1)
        return normalized_tensor_array

class BertSemantic(nn.Module):
    def __init__(self, output_size, bert):
        super(BertSemantic, self).__init__()
        self.bert = bert
        for name, para in self.bert.named_parameters():
            para.requires_grad_(False)
        self.fc = nn.Linear(768,output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        semantic_outputs = outputs[0][:,0,:]     #[0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果
        output = self.fc(semantic_outputs)
        return output

class BertGNNGru(nn.Module):
    def __init__(self, bert_with_layer_size, news_size, bert, num_heads):
        super(BertGNNGru, self).__init__()
        self.out_bert_size = news_size
        self.bertwithlayer = BertWithLayer(output_size= bert_with_layer_size, bert=bert)
        self.bertwithoutlayer = BertWithoutLayer(output_size= bert_with_layer_size, bert=bert)
        self.bertpure = BertSemantic(bert_with_layer_size, bert=bert)
        self.attention = nn.MultiheadAttention(news_size,num_heads=1)
        self.Q = nn.Linear(news_size, 10)
        self.K = nn.Linear(news_size, 10)
        self.V = nn.Linear(news_size, 10)
        self.linear = nn.Linear(10, 1)
        self.self_attention = nn.MultiheadAttention(50, num_heads)
        self.att = nn.Linear(20, 10)

    def forward(self, news_data, masks_data, corporate_data, corporate_masks_data):
        '''
        news start
        '''
        #

        output_tensor = self.bertwithlayer(news_data,masks_data)
        #print(output_tensor.shape)
        '''
        这段我给去掉了，后面看看能不能用
        '''
        output_cor_tensor = self.bertwithoutlayer(corporate_data, corporate_masks_data)
        cat_tensor = torch.cat((output_tensor, output_cor_tensor), dim=1)
        output_tensor= torch.tanh_(self.att(cat_tensor))
        # print(output_tensor)
        output = self.linear(output_tensor)
        return output, output_tensor