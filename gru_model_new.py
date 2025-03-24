import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertLayer
import math
from torch.autograd import Variable
import pandas as pd
import numpy as np
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim

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

                dense_tensor = last_hidden_state[:, i, :].to('cuda')
                # 计算每个向量的各位加和
                sum_elements = torch.sum(dense_tensor, dim=1)

                # 找到非全零的向量
                non_zero_indices = torch.nonzero(sum_elements).squeeze()

                process_tensor = torch.randn(dense_tensor.shape[0], 10)

                # 对非全零的向量进行 softmax
                softmaxed_tensor_array = torch.zeros_like(process_tensor).to('cuda')
                if non_zero_indices.numel() > 0:
                    softmaxed_tensor_array[non_zero_indices] = nn.functional.softmax(self.fc(dense_tensor[non_zero_indices]), dim=-1)

                # 最后将全零向量位置保持为零
                softmaxed_tensor_array[sum_elements == 0] = 0
                dense_tensor_soft = softmaxed_tensor_array
            else:
                last_hidden_state[:, i, :] = self.layer(last_hidden_state[:, i, :].unsqueeze(1) + last_hidden_state[:, i - 1, :].unsqueeze(1))[0].squeeze(1)
                dense_tensor = last_hidden_state[:, i, :].to('cuda')
                # 计算每个向量的各位加和
                sum_elements = torch.sum(dense_tensor, dim=1)

                # 找到非全零的向量
                non_zero_indices = torch.nonzero(sum_elements).squeeze()
                process_tensor = torch.randn(dense_tensor.shape[0], 10)

                # 对非全零的向量进行 softmax
                softmaxed_tensor_array = torch.zeros_like(process_tensor).to('cuda')
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

            # 对非全零的向量进行 softmax
            tensor_array = torch.zeros_like(dense_tensor)
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

class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.attention = nn.Linear(2 * hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1)).to('cuda')

        gate_x = self.x2h(x)
        hidden = hidden.to('cuda')
        gate_h = self.h2h(hidden)
        #print(gate_x.shape)
        #gate_x = gate_x.squeeze()
        #gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        # i_i = torch.tanh(i_i)
        # h_i = torch.tanh(h_i)
        attention_tmp = torch.cat((i_i, h_i), dim=-1)
        #         print(i_i.size())
        #         print(attention_tmp.size())

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(self.attention(attention_tmp))
        # inputgate = torch.sigmoid(self.attention(attention_tmp))
        newgate = torch.tanh_(i_n + (resetgate * h_n))

        #         print(resetgate.size())
        #         print(inputgate.size())
        #         print(newgate.size())

        hy = newgate + inputgate * (hidden - newgate)
        # print(hy.size())

        return hy


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, bias=True):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # print(x.shape,"x.shape")100, 28, 28
        '''
        :param x:
        :return:
        这里先不使用
        '''
        # if torch.cuda.is_available():
        #     h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        # else:
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []

        hn = h0[0, :, :]

        # print(x.size())
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)

        # out = outs[-1].squeeze()

        '''
        GPU被去掉
        '''
        outs = torch.tensor([item.cpu().detach().numpy() for item in outs])#.cuda()
        # out = torch.Tensor(outs)
        # out.size() --> 100, 10
        # print(outs.shape)
        out = outs.permute(1, 0, 2)
        return out

class GRUTraCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUTraCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.attention = nn.Linear(2 * hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1)).to('cuda')

        gate_x = self.x2h(x)
        hidden = hidden.to('cuda')
        gate_h = self.h2h(hidden)
        #print(gate_x.shape)
        #gate_x = gate_x.squeeze()
        #gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        attention_tmp = torch.cat((i_i, h_i), dim=-1)
        #         print(i_i.size())
        #         print(attention_tmp.size())

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        #         print(resetgate.size())
        #         print(inputgate.size())
        #         print(newgate.size())

        hy = newgate + inputgate * (hidden - newgate)
        # print(hy.size())

        return hy


class GRUTraModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, bias=True):
        super(GRUTraModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.gru_cell = GRUTraCell(input_dim, hidden_dim, layer_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # print(x.shape,"x.shape")100, 28, 28
        '''
        :param x:
        :return:
        这里先不使用
        '''
        # if torch.cuda.is_available():
        #     h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        # else:
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []

        hn = h0[0, :, :]

        # print(x.size())
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)

        # out = outs[-1].squeeze()

        '''
        GPU被去掉
        '''
        outs = torch.tensor([item.cpu().detach().numpy() for item in outs])#.cuda()
        # out = torch.Tensor(outs)
        # out.size() --> 100, 10
        # print(outs.shape)
        out = outs.permute(1, 0, 2)
        return out

class DeepGru(nn.Module):
    def __init__(self, input_size):
        super(DeepGru, self).__init__()
        self.gru_1 = GRUModel(input_size, 16, 1)
        self.gru_2 = GRUModel(16, 16, 1)
        self.gru_3 = GRUModel(16, 16, 1)
        self.layer_norm = nn.LayerNorm(16)


        self.linear_3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        use_cuda = 1
        device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
        # h0 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1 = self.gru_1(x).to(device)
        # out_1 = self.layer_norm(out_1)
        # print(out_1.size())
        out_1 = self.dropout(out_1)
        # h1 = torch.zeros(1, x.size(0), 512).to(device)
        out_2 = 0.7 * out_1 + 0.3 * self.gru_2(out_1).to(device)
        # out_2 = self.layer_norm(out_2)
        out_2 = self.dropout(out_2)
        # h2 = torch.zeros(1, x.size(0), 256).to(device)
        out_3 = 0.7 * out_2 + 0.3 * self.gru_3(out_2).to(device)
        # out_3 = self.layer_norm(out_3)
        out_3 = self.dropout(out_3).to('cuda')
        out = self.linear_3(out_3[:, -1, :])

        out = torch.sigmoid_(out)
        return out

class DeepTraGru(nn.Module):
    def __init__(self, input_size):
        super(DeepTraGru, self).__init__()
        self.gru_1 = GRUTraModel(input_size, 1024, 1)
        self.gru_2 = GRUTraModel(1024, 512, 1)
        self.gru_3 = GRUTraModel(512, 256, 1)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        use_cuda = 1
        device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
        h0 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1 = self.gru_1(x)
        # print(out_1.size())
        out_1 = self.dropout(out_1)
        h1 = torch.zeros(1, x.size(0), 512).to(device)
        out_2 = self.gru_2(out_1)
        out_2 = self.dropout(out_2)
        h2 = torch.zeros(1, x.size(0), 256).to(device)
        out_3 = self.gru_3(out_2)
        out_3 = self.dropout(out_3).to('cuda')
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out = self.linear_3(out_5)
        return out

class BertGNNGru(nn.Module):
    def __init__(self, bert_with_layer_size, news_size, in_channels, hidden_channels, out_channels, num_types, metadata, input_size, bert, num_heads):
        super(BertGNNGru, self).__init__()
        self.out_bert_size = news_size
        self.bertwithlayer = BertWithLayer(output_size= bert_with_layer_size, bert=bert)
        self.bertwithoutlayer = BertWithoutLayer(output_size= bert_with_layer_size, bert=bert)
        self.bertpure = BertSemantic(bert_with_layer_size, bert=bert)
        self.attention = nn.MultiheadAttention(news_size,num_heads=1)
        # self.graphnn = HANModel(in_channels= in_channels,hidden_channels= hidden_channels,out_channels= out_channels,num_types= num_types,metadata= metadata)
        self.deepgru = DeepGru(input_size=input_size)
        self.Q = nn.Linear(news_size, 10)
        self.K = nn.Linear(news_size, 10)
        self.V = nn.Linear(news_size, 10)
        self.linear = nn.Linear(77, 50)
        self.self_attention = nn.MultiheadAttention(50, num_heads)

    def forward(self, news_data, masks_data, policy_data, policy_masks_data, graph_batch, basic_data):
        '''
        news start
        '''
        batch_size_news, num_items_news, total_news = news_data.shape
        batch_size_masks, num_items_masks, total_masks = masks_data.shape
        # batch_size_cors, num_items_cors, total_cors = corporate_data.shape
        # batch_size_cormasks, num_items_cormasks, total_cormasks = corporate_masks_data.shape

        news_data = news_data.view(batch_size_news, num_items_news, 10, 512)
        masks_data = masks_data.view(batch_size_masks, num_items_masks, 10, 512)
        # corporate_data = corporate_data.view(batch_size_cors, num_items_cors, 10, 512)
        # corporate_masks_data = corporate_masks_data.view(batch_size_cormasks, num_items_cormasks, 10, 512)

        batch_size_news, num_items_news, max_rows_news, features_news = news_data.shape
        batch_size_masks, num_items_masks, max_rows_masks, features_masks = masks_data.shape
        # batch_size_cors, num_items_cors, max_rows_cors, features_cors = corporate_data.shape
        # batch_size_cormasks, num_items_cormasks, max_rows_cormasks, features_cormasks = corporate_masks_data.shape
        #
        rearranged_ids_tensor = news_data.reshape(batch_size_news * num_items_news * max_rows_news, features_news)
        rearranged_mask_tensor = masks_data.reshape(batch_size_masks * num_items_masks * max_rows_masks, features_masks)
        # rearranged_cors_tensor = corporate_data.view(batch_size_cors * num_items_cors * max_rows_cors, features_cors)
        # rearranged_cormasks_tensor = corporate_masks_data.view(batch_size_cormasks * num_items_cormasks * max_rows_cormasks, features_cormasks)

        output_tensor = self.bertwithlayer(rearranged_ids_tensor,rearranged_mask_tensor)
        #print(output_tensor.shape)
        '''
        这段我给去掉了，后面看看能不能用
        '''
        #output_cor_tensor = self.bertwithoutlayer(rearranged_cors_tensor, rearranged_cormasks_tensor)
        #output_tensor = self.attention(output_tensor, output_cor_tensor, output_tensor)

        output_tensor = output_tensor.view(batch_size_news * num_items_news, max_rows_news, self.out_bert_size)
        # 找到非零元素的个数
        nonzero_counts = torch.sum(output_tensor != 0, dim=1)
        #print(nonzero_counts.shape)

        # 将零元素置为1，以免除数为0
        nonzero_counts[nonzero_counts == 0] = 1

        # 求每行的和
        row_sums = torch.sum(output_tensor, dim=1)

        # 对每个矩阵做每行求和再除以非零行的个数
        result = row_sums.float() / nonzero_counts.float()
        result = result.view(batch_size_news * num_items_news, self.out_bert_size)
        result = result.unsqueeze(1)

        que = self.Q(output_tensor)
        key = self.K(result)
        val = self.V(output_tensor)
        #key_ex = key.expand(-1, que.size(1), -1)
        # 计算注意力数值
        attention_scores = torch.matmul(que, key.transpose(-2, -1))  # 在最后两个维度上做点积

        # 对每行的注意力数值进行 softmax
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-2)

        # 将注意力权重乘以原始的 5*10 矩阵
        weighted_matrix = attention_weights * val

        summed_matrix = torch.sum(weighted_matrix, dim=1, keepdim=True)

        news_matrix = torch.cat((summed_matrix, result), dim=2)

        news_matrix = torch.squeeze(news_matrix, dim=1)
        news_matrix = news_matrix.view(batch_size_news, num_items_news, -1)
        '''
        news end
        '''
        '''
        policy start
        '''
        policy_data = torch.squeeze(policy_data, dim=1)
        policy_masks_data = torch.squeeze(policy_masks_data, dim=1)
        batch_size_policies, num_items_policies, features_policies = policy_data.shape
        batch_size_policy_masks, num_items_policy_masks, features_policy_masks = policy_masks_data.shape

        rearranged_policies_tensor = policy_data.view(batch_size_policies * num_items_policies, features_policies)
        rearranged_policy_mask_tensor = policy_masks_data.view(batch_size_policy_masks * num_items_policy_masks, features_policy_masks)

        output_policy_tensor = self.bertpure(rearranged_policies_tensor, rearranged_policy_mask_tensor)

        output_policy_tensor = output_policy_tensor.view(batch_size_policies, num_items_policies, self.out_bert_size)
        '''
        policy_end
        '''
        '''
        graph start
        '''
        batch_size_graph, num_items_graph, graph_size = graph_batch.shape
        total_graph_tensor = torch.cat((basic_data, news_matrix, output_policy_tensor), dim=2)
        graph_tensor = self.linear(total_graph_tensor)
        graph_total_tensor = torch.cat((graph_tensor, graph_tensor, graph_tensor), dim=2)
        graph_total_tensor[graph_batch == 0] = 0
        graph_total_tensor = graph_total_tensor.view(batch_size_graph * num_items_graph * 3, 50)
        graph_total_tensor = self.self_attention(graph_total_tensor, graph_total_tensor, graph_total_tensor)
        graph_total_tensor = graph_total_tensor[0]
        graph_total_tensor = graph_total_tensor.view(batch_size_graph * num_items_graph * 3, 5, 10)
        # 在第二个维度上求和
        sum_tensor = torch.sum(graph_total_tensor, dim=1)

        nonzero_counts = torch.count_nonzero(graph_total_tensor, dim=1)

        # 将零值替换为 1，避免除零错误
        nonzero_counts = nonzero_counts.to('cuda')
        nonzero_counts = torch.where(nonzero_counts == 0, torch.tensor(1).to('cuda'), nonzero_counts)

        # 将 sum_tensor 中每个元素除以对应列的非零元素的个数
        graph_tensor = sum_tensor / nonzero_counts

        # list_1 = []
        # list_2 = []
        # length_2 = 0
        # length_1 = 0
        # cnt = 0
        # concat_tensor = torch.randn(1, 30)
        #
        # for group_data in graph_batch:
        #     for subgroup_data in group_data:
        #         #for graph in subgroup_data:
        #         if subgroup_data[f'Policy_x'].size(0) == 0:
        #             continue
        #         else:
        #             subgroup_data[f'Policy_x'] = output_policy_tensor
        #         if subgroup_data[f'News_x'].size(0) == 0:
        #             continue
        #         else:
        #             subgroup_data[f'News_x'] = news_matrix
        #
        #         output_graph_tensor = self.graphnn(subgroup_data)
        #         # print(output_graph_tensor.shape)
        #         if cnt == 0:
        #             concat_tensor = output_graph_tensor
        #             cnt = 1
        #         else:
        #             concat_tensor = torch.cat((concat_tensor, output_graph_tensor), dim=1)
        #     # print(concat_tensor.shape)
        #     list_2.append(concat_tensor)
        #     cnt = 0
        #     length_2 = len(group_data)
        #     # tmp_df = pd.DataFrame([tensor.flatten().numpy() for tensor in list_2], dtype=np.float32)
        #     tmp_df = torch.stack(list_2)
        #     tmp_df = tmp_df.cuda()
        #     # print(tmp_df.shape)
        #     list_1.append(tmp_df)
        #     list_2 = []
        # length_1 = len(graph_batch)
        # # graph_tensor = pd.DataFrame([tensor.flatten().numpy() for tensor in list_1], dtype=np.float32)
        # graph_tensor = torch.stack(list_1)
        # #print(graph_tensor.shape)
        # graph_tensor = graph_tensor.unsqueeze(0)
        # #graph_tensor = graph_tensor.view(length_1, length_2, -1)
        '''
        graph end
        '''
        '''
        gru start
        '''
        # print(graph_tensor.shape)
        graph_tensor = graph_tensor.to('cuda')
        graph_tensor = graph_tensor.squeeze()
        # print(graph_tensor.shape)
        graph_tensor = graph_tensor.view(batch_size_graph, num_items_graph, -1)
        total_tensor = torch.cat((basic_data, news_matrix, output_policy_tensor, graph_tensor), dim=2)
        # print(total_tensor.shape)
        # print(total_tensor.shape)
        output = self.deepgru(total_tensor)

        return output

class testmodel(nn.Module):
    def __init__(self, input_size):
        super(testmodel, self).__init__()
        self.deepgru = DeepGru(input_size=40)
        self.basic_fc = nn.Linear(input_size, 30)
        self.fc1 = nn.Linear(768, 192)
        self.fc2 = nn.Linear(192, 48)
        self.fc3 = nn.Linear(48, 10)
        self.self_attention = nn.MultiheadAttention(40, num_heads=1)
        self.fc4 = nn.Linear(80, 40)

    def forward(self, basic_data, news_data):
        '''
        news start
        '''
        batch_size_basic, num_items_basic, total_basic = basic_data.shape
        batch_size_news, num_items_news, total_news = news_data.shape
        fc1 = self.fc1(news_data)
        tan1 = F.tanh(fc1)
        fc2 = self.fc2(tan1)
        tan2 = F.tanh(fc2)
        fc3 = self.fc3(tan2)
        news_ten = F.tanh(fc3)
        data_ten = self.basic_fc(basic_data)

        cat_data = torch.cat((data_ten, news_ten), dim=2)

        graph_total_tensor = cat_data.view(batch_size_news * num_items_news, -1)
        graph_total_tensor = self.self_attention(graph_total_tensor, graph_total_tensor, graph_total_tensor)
        graph_total_tensor = graph_total_tensor[0]
        graph_total_tensor = graph_total_tensor.view(batch_size_news ,num_items_news, -1)
        total_data = torch.cat((cat_data, graph_total_tensor), dim=2)
        total_data_0 = self.fc4(total_data)
        output = self.deepgru(total_data_0)

        return output


class StockTransformer(nn.Module):
    def __init__(self, input_dim, output_dim=1, num_layers=6, num_heads=8, hidden_dim=512, dropout=0.1):
        super(StockTransformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.fc(x.mean(dim=1))  # Global average pooling

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=1, num_layers=1):
        super(StockLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])  # Get output from last time step

        return out


class StockGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=1, num_layers=3):
        super(StockGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.gru(x, h0)

        out = self.fc(out)  # Get output from last time step

        return out

class DeepOldGru(nn.Module):
    def __init__(self, input_size):
        super(DeepOldGru, self).__init__()
        self.gru_1 = nn.GRU(input_size, 1024, 1)
        self.gru_2 = nn.GRU(1024, 512, 1)
        self.gru_3 = nn.GRU(512, 256, 1)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        use_cuda = 1
        device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
        h0 = torch.zeros(1, x.size(1), 1024).to(device)
        out_1, _ = self.gru_1(x, h0)
        # print(out_1.size())
        out_1 = self.dropout(out_1)
        h1 = torch.zeros(1, x.size(1), 512).to(device)
        out_2, _ = self.gru_2(out_1, h1)
        out_2 = self.dropout(out_2)
        h2 = torch.zeros(1, x.size(1), 256).to(device)
        out_3, _ = self.gru_3(out_2, h2)
        out_3 = self.dropout(out_3).to(device)
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out = self.linear_3(out_5)
        return out

class testmodelnew(nn.Module):
    def __init__(self, input_size):
        super(testmodelnew, self).__init__()
        self.deepoldgru = DeepGru(input_size=input_size)
        self.fc1 = nn.Linear(768, 192)
        self.fc2 = nn.Linear(192, 48)
        self.fc3 = nn.Linear(48, 10)
        self.fc5 = nn.Linear(768, 192)
        self.fc6 = nn.Linear(192, 48)
        self.fc7 = nn.Linear(48, 10)
        self.self_attention = nn.MultiheadAttention(40, num_heads=1)
        self.fc4 = nn.Linear(80, 40)

    def forward(self, basic_data, news_data, policy_data):
        '''
        news start
        '''
        batch_size_basic, num_items_basic, total_basic = basic_data.shape
        batch_size_news, num_items_news, total_news = news_data.shape
        fc1 = self.fc1(news_data)
        tan1 = torch.tanh_(fc1)
        fc2 = self.fc2(tan1)
        tan2 = torch.tanh_(fc2)
        fc3 = self.fc3(tan2)
        news_ten = torch.tanh_(fc3)

        fc5 = self.fc5(policy_data)
        tan5 = torch.tanh_(fc5)
        fc6 = self.fc6(tan5)
        tan6 = torch.tanh_(fc6)
        fc7 = self.fc3(tan6)
        policy_ten = torch.tanh_(fc7)

        cat_data = torch.cat((basic_data, news_ten, policy_ten), dim=2)

        output = self.deepoldgru(cat_data)

        return output

class testmodelwithoutbert(nn.Module):
    def __init__(self, input_size):
        super(testmodelwithoutbert, self).__init__()
        self.deepoldgru = DeepGru(input_size=input_size)
        self.fc1 = nn.Linear(768, 192)
        self.fc2 = nn.Linear(192, 48)
        self.fc3 = nn.Linear(48, 10)
        self.fc5 = nn.Linear(768, 192)
        self.fc6 = nn.Linear(192, 48)
        self.fc7 = nn.Linear(48, 10)
        self.self_attention = nn.MultiheadAttention(40, num_heads=1)
        self.fc4 = nn.Linear(80, 40)

    def forward(self, basic_data):
        '''
        news start
        '''
        batch_size_basic, num_items_basic, total_basic = basic_data.shape

        output = self.deepoldgru(basic_data)

        return output

class testmodelwithoutgnn(nn.Module):
    def __init__(self, input_size):
        super(testmodelwithoutgnn, self).__init__()
        self.deepoldgru = DeepGru(input_size=input_size)
        self.fc1 = nn.Linear(768, 192)
        self.fc2 = nn.Linear(192, 48)
        self.fc3 = nn.Linear(48, 10)
        self.fc5 = nn.Linear(768, 192)
        self.fc6 = nn.Linear(192, 48)
        self.fc7 = nn.Linear(48, 10)
        self.self_attention = nn.MultiheadAttention(40, num_heads=1)
        self.fc4 = nn.Linear(80, 40)

    def forward(self, basic_data, news_data, policy_data):
        '''
        news start
        '''
        fc1 = self.fc1(news_data)
        tan1 = torch.tanh_(fc1)
        fc2 = self.fc2(tan1)
        tan2 = torch.tanh_(fc2)
        fc3 = self.fc3(tan2)
        news_ten = torch.tanh_(fc3)

        fc5 = self.fc5(policy_data)
        tan5 = torch.tanh_(fc5)
        fc6 = self.fc6(tan5)
        tan6 = torch.tanh_(fc6)
        fc7 = self.fc3(tan6)
        policy_ten = torch.tanh_(fc7)

        cat_data = torch.cat((basic_data, news_ten, policy_ten), dim=2)

        output = self.deepoldgru(cat_data)

        return output

class DeepNewGru(nn.Module):
    def __init__(self, input_size):
        super(DeepNewGru, self).__init__()
        self.gru_1 = nn.GRU(input_size, 256, 1)
        self.gru_2 = nn.GRU(256, 128, 1)
        self.linear_1 = nn.Linear(128, 64)
        self.linear_2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        use_cuda = 1
        device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
        h0 = torch.zeros(1, x.size(1), 256).to(device)
        out_1, _ = self.gru_1(x, h0)
        # print(out_1.size())
        out_1 = self.dropout(out_1)
        h1 = torch.zeros(1, x.size(1), 128).to(device)
        out_2, _ = self.gru_2(out_1, h1)
        out_2 = self.dropout(out_2)
        out_3 = self.linear_1(out_2[:, -1, :])
        out = self.linear_2(out_3)
        return out

class testmodelwithattention(nn.Module):
    def __init__(self, input_size):
        super(testmodelwithattention, self).__init__()
        self.deepgru = DeepGru(input_size=input_size)
        self.deepgrunew = DeepTraGru(input_size=input_size)
        self.fc1 = nn.Linear(768, 192)
        self.fc2 = nn.Linear(192, 48)
        self.fc3 = nn.Linear(48, 10)
        self.fc5 = nn.Linear(768, 192)
        self.fc6 = nn.Linear(192, 48)
        self.fc7 = nn.Linear(48, 10)
        self.self_attention = nn.MultiheadAttention(40, num_heads=1)
        self.fc4 = nn.Linear(80, 40)

    def forward(self, basic_data, news_data, policy_data):
        '''
        news start
        '''
        batch_size_basic, num_items_basic, total_basic = basic_data.shape
        batch_size_news, num_items_news, total_news = news_data.shape
        fc1 = self.fc1(news_data)
        tan1 = torch.tanh_(fc1)
        fc2 = self.fc2(tan1)
        tan2 = torch.tanh_(fc2)
        fc3 = self.fc3(tan2)
        news_ten = torch.tanh_(fc3)

        fc5 = self.fc5(policy_data)
        tan5 = torch.tanh_(fc5)
        fc6 = self.fc6(tan5)
        tan6 = torch.tanh_(fc6)
        fc7 = self.fc3(tan6)
        policy_ten = torch.tanh_(fc7)

        cat_data = torch.cat((basic_data, news_ten, policy_ten), dim=2)

        output = self.deepgrunew(cat_data)

        return output

class CTTS(nn.Module):
    def __init__(self, input_dim, output_dim = 1, conv_channels = 4, conv_kernel_size = 16, num_layers=1, d_model=4, num_heads = 1, dropout = 0.1):
        super(CTTS, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(conv_channels, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        x = self.conv(x)  # (batch_size, conv_channels, new_seq_len)
        x = x.permute(2, 0, 1)  # (new_seq_len, batch_size, conv_channels)

        # Transformer expects (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)

        # Take the last output from the sequence and pass through linear layer
        x = self.fc(x[-1])
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(z))  # Assuming output in [0, 1] range
        return x_recon


class FactorVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim = 16, latent_dim=1, output_dim=1):
        super(FactorVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        x_recon = x_recon[:, -1 ,:]
        return x_recon

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class NGCU(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(NGCU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Recurrent neural network layer
        self.rnn = nn.GRUCell(input_dim, hidden_dim)

        # Linear layers for computing causality scores
        self.linear_influence = nn.Linear(hidden_dim, hidden_dim)
        self.linear_gate = nn.Linear(hidden_dim, hidden_dim)
        self.linear_causality_score = nn.Linear(hidden_dim, 1)
        self.device = 'cuda'

    def forward(self, x):
        # x: input tensor of shape (batch_size, input_dim)
        # prev_hidden: previous hidden state of shape (batch_size, hidden_dim)

        h0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim))

        outs = []

        hn = h0[0, :, :].to(self.device)

        # print(x.size())
        for seq in range(x.size(1)):
            # Recurrent neural network step
            hn = self.rnn(x[:, seq, :], hn)

            # Compute causality scores
            influence_score = torch.tanh(self.linear_influence(hn))
            gate_score = torch.sigmoid(self.linear_gate(hn))
            causality_score = torch.sigmoid(self.linear_causality_score(influence_score * gate_score))
            outs.append(causality_score)

        # out = outs[-1].squeeze()

        '''
        GPU被去掉
        '''
        outs = torch.tensor([item.cpu().detach().numpy() for item in outs])#.cuda()
        # out = torch.Tensor(outs)
        # out.size() --> 100, 10
        # print(outs.shape)
        out = outs.permute(1, 0, 2)
        out = out[:, -1, :]
        return out


class CNNTimeSeries(nn.Module):
    def __init__(self, input_dim, output_dim = 1, conv_channels = 4, conv_kernel_size = 16, dropout = 0.1):
        super(CNNTimeSeries, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # Calculate the output size after convolution
        self.conv_out_size = conv_channels * (((input_dim - conv_kernel_size + 1) // 2) // conv_kernel_size)

        self.fc = nn.Sequential(
            nn.Linear(28, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        x = self.conv(x)  # (batch_size, conv_channels, new_seq_len)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

class GRUDifferent(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, bias=True):
        super(GRUDifferent, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.gru_cell_1 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_2 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_3 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_4 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_5 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_6 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_7 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_8 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_9 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_10 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_11 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_12 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_13 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_14 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_15 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_16 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_17 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_18 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_19 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_20 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_21 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_22 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_23 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_24 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_25 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_26 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_27 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_28 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_29 = GRUCell(input_dim, hidden_dim)
        self.gru_cell_30 = GRUCell(input_dim, hidden_dim)


    def forward(self, x):

        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # print(x.shape,"x.shape")100, 28, 28
        '''
        :param x:
        :return:
        这里先不使用
        '''
        # if torch.cuda.is_available():
        #     h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        # else:
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []

        hn = h0[0, :, :]

        # print(x.size())

        h1 = self.gru_cell_1(x[:, 0, :], hn)
        outs.append(h1)
        h2 = self.gru_cell_2(x[:, 1, :], h1)
        outs.append(h2)
        h3 = self.gru_cell_3(x[:, 2, :], h2)
        outs.append(h3)
        h4 = self.gru_cell_4(x[:, 3, :], h3)
        outs.append(h4)
        h5 = self.gru_cell_1(x[:, 4, :], h4)
        outs.append(h5)
        h6 = self.gru_cell_1(x[:, 5, :], h5)
        outs.append(h6)
        h7 = self.gru_cell_1(x[:, 6, :], h6)
        outs.append(h7)
        h8 = self.gru_cell_1(x[:, 7, :], h7)
        outs.append(h8)
        h9 = self.gru_cell_1(x[:, 8, :], h8)
        outs.append(h9)
        h10 = self.gru_cell_1(x[:, 9, :], h9)
        outs.append(h10)
        h11 = self.gru_cell_1(x[:, 10, :], h10)
        outs.append(h11)
        h12 = self.gru_cell_1(x[:, 11, :], h11)
        outs.append(h12)
        h13 = self.gru_cell_1(x[:, 12, :], h12)
        outs.append(h13)
        h14 = self.gru_cell_1(x[:, 13, :], h13)
        outs.append(h14)
        h15 = self.gru_cell_1(x[:, 14, :], h14)
        outs.append(h15)
        h16 = self.gru_cell_1(x[:, 15, :], h15)
        outs.append(h16)
        h17 = self.gru_cell_1(x[:, 16, :], h16)
        outs.append(h17)
        h18 = self.gru_cell_1(x[:, 17, :], h17)
        outs.append(h18)
        h19 = self.gru_cell_1(x[:, 18, :], h18)
        outs.append(h19)
        h20 = self.gru_cell_1(x[:, 19, :], h19)
        outs.append(h20)
        h21 = self.gru_cell_1(x[:, 20, :], h20)
        outs.append(h21)
        h22 = self.gru_cell_1(x[:, 21, :], h21)
        outs.append(h22)
        h23 = self.gru_cell_1(x[:, 22, :], h22)
        outs.append(h23)
        h24 = self.gru_cell_1(x[:, 23, :], h23)
        outs.append(h24)
        h25 = self.gru_cell_1(x[:, 24, :], h24)
        outs.append(h25)
        h26 = self.gru_cell_1(x[:, 25, :], h25)
        outs.append(h26)
        h27 = self.gru_cell_1(x[:, 26, :], h26)
        outs.append(h27)
        h28 = self.gru_cell_1(x[:, 27, :], h27)
        outs.append(h28)
        h29 = self.gru_cell_1(x[:, 28, :], h28)
        outs.append(h29)
        h30 = self.gru_cell_1(x[:, 29, :], h29)
        # out = outs[-1].squeeze()
        outs.append(h30)

        '''
        GPU被去掉
        '''
        outs = torch.tensor([item.cpu().detach().numpy() for item in outs])#.cuda()
        # out = torch.Tensor(outs)
        # out.size() --> 100, 10
        # print(outs.shape)
        out = outs.permute(1, 0, 2)
        return out

class DeepDifferentGru(nn.Module):
    def __init__(self, input_size):
        super(DeepDifferentGru, self).__init__()
        self.gru_1 = GRUDifferent(input_size, 16, 3)
        self.gru_2 = StockGRU(16, 16, 16)

        self.linear_3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        use_cuda = 1
        device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
        # h0 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1 = self.gru_1(x).to(device)
        # out_1 = self.layer_norm(out_1)
        # print(out_1.size())
        out_1 = self.dropout(out_1)

        # h1 = torch.zeros(1, x.size(0), 512).to(device)
        # print(out_1)
        out_2 = 0.7 * out_1 + 0.3 * self.gru_2(out_1).to(device)
        # out_2 = self.layer_norm(out_2)
        out_2 = self.dropout(out_2)

        out = self.linear_3(out_2[:, -1, :])
        out = torch.sigmoid_(out)
        return out

class TRA(nn.Module):
    def __init__(self, input_size, num_states=1, hidden_size=8, tau=1.0, src_info="LR_TPE"):
        super().__init__()
        self.num_states = num_states
        self.tau = tau
        self.src_info = src_info

        if num_states > 1:
            self.router = nn.LSTM(
                input_size=num_states,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size + input_size, num_states)

        self.predictors = nn.Linear(input_size, num_states)

    def forward(self, hidden):
        batch_size = hidden.size(0)
        hist_loss = torch.zeros((batch_size, 1, self.num_states)).to(hidden.device)  # Initialize hist_loss as zeros

        preds = self.predictors(hidden)

        if self.num_states == 1:
            return preds.squeeze(-1)  # Only return final prediction tensor

        router_out, _ = self.router(hist_loss)
        if "LR" in self.src_info:
            latent_representation = hidden
        else:
            latent_representation = torch.randn(hidden.shape).to(hidden.device)
        if "TPE" in self.src_info:
            temporal_pred_error = router_out[:, -1]
        else:
            temporal_pred_error = torch.randn(router_out[:, -1].shape).to(hidden.device)

        out = self.fc(torch.cat([temporal_pred_error, latent_representation], dim=-1))
        prob = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=False)

        if self.training:
            final_pred = (preds * prob).sum(dim=-1)
        else:
            final_pred = preds[range(len(preds)), prob.argmax(dim=-1)]

        return final_pred  # Only return final prediction tensor


class LSTM_HA(nn.Module):
    '''
    Here we employ the attention to LSTM to capture the time series traits more efficiently.
    '''

    def __init__(self, in_features, hidden_dim=64, output_dim=64, n_heads=4):
        super(LSTM_HA, self).__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=output_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        attn_output, _ = self.attention(outputs, outputs, outputs)
        return attn_output.relu()

class CAAN(nn.Module):
    """Cross-Asset Attention"""

    def __init__(self, input_dim, output_dim, hidden_dim=64, n_heads=4):
        super(CAAN, self).__init__()

        self.W_q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, output_dim)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        queries, keys, values = self.W_q(x), self.W_k(x), self.W_v(x)
        attn_output, _ = self.attention(queries, keys, values)
        scores = self.W_s(attn_output)
        return scores.relu()

class AlphaStock(nn.Module):
    def __init__(self, dim_in, dim_enc=32, n_heads=2, negative_slope=0.01):
        super(AlphaStock, self).__init__()

        self.lstm_ha = LSTM_HA(dim_enc, dim_enc, dim_enc, n_heads)
        self.caan = CAAN(dim_enc, dim_enc, dim_enc, n_heads)

        # Dense layers for managing network inputs and outputs
        self.input_fc = nn.Linear(dim_in, dim_enc)
        self.out_fc = nn.Linear(dim_enc, 1)

        self.leakyrelu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        e = self.input_fc(x)
        e = self.leakyrelu(e)
        e = self.lstm_ha(e)
        e = self.caan(e)

        return self.out_fc(e[:, -1, :]).squeeze(-1)

class MSU(nn.Module):
    def __init__(self, in_features, window_len=30, hidden_dim=16):
        super(MSU, self).__init__()
        self.in_features = in_features
        self.window_len = window_len
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim)
        self.attn1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn2 = nn.Linear(hidden_dim, 1)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, X):
        """
        :X: [batch_size(B), window_len(L), in_features(I)]
        :return: Parameters: [batch, 2]
        """
        X = X.permute(1, 0, 2)

        outputs, (h_n, c_n) = self.lstm(X)  # lstm version
        H_n = h_n.repeat((self.window_len, 1, 1))
        scores = self.attn2(torch.tanh(self.attn1(torch.cat([outputs, H_n], dim=2))))  # [L, B*N, 1]
        scores = scores.squeeze(2).transpose(1, 0)  # [B*N, L]
        attn_weights = torch.softmax(scores, dim=1)
        outputs = outputs.permute(1, 0, 2)  # [B*N, L, H]
        attn_embed = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)
        embed = torch.relu(self.bn1(self.linear1(attn_embed)))
        parameters = self.linear2(embed)
        # return parameters[:, 0], parameters[:, 1]   # mu, sigma
        return parameters.squeeze(-1)


class Mamba(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, d_model: int = 32, n_layer: int = 1, d_state: int = 16,
                 expand: int = 2, dt_rank: Union[int, str] = 'auto', d_conv: int = 4, conv_bias: bool = True,
                 bias: bool = False):
        """Full Mamba model adjusted for stock prediction."""
        super().__init__()

        # Initialize ModelArgs internally
        self.args = self.ModelArgs(d_model, n_layer, d_state, expand, dt_rank, d_conv, conv_bias, bias)

        # Embedding layer for input features
        self.input_fc = nn.Linear(input_dim, self.args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(self.args) for _ in range(self.args.n_layer)])
        self.norm_f = RMSNorm(self.args.d_model)
        self.output_fc = nn.Linear(self.args.d_model, output_dim)

    def forward(self, x):
        """
        Args:
            x (tensor): shape (b, l, input_dim)

        Returns:
            predictions: shape (b, l, output_dim)
        """
        x = self.input_fc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        predictions = self.output_fc(x)
        return predictions[:,-1,:]

    @dataclass
    class ModelArgs:
        d_model: int
        n_layer: int
        d_state: int = 16
        expand: int = 2
        dt_rank: Union[int, str] = 'auto'
        d_conv: int = 4
        conv_bias: bool = True
        bias: bool = False

        def __post_init__(self):
            self.d_inner = int(self.expand * self.d_model)
            if self.dt_rank == 'auto':
                self.dt_rank = math.ceil(self.d_model / 16)


class ResidualBlock(nn.Module):
    def __init__(self, args: Mamba.ModelArgs):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output


class MambaBlock(nn.Module):
    def __init__(self, args: Mamba.ModelArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1D, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]


class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks):
        super(BlockDiagonal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks

        if in_features % num_blocks != 0 or out_features % num_blocks != 0:
            raise ValueError("in_features and out_features must be divisible by num_blocks")

        block_in_features = in_features // num_blocks
        block_out_features = out_features // num_blocks
        self.blocks = nn.ModuleList([nn.Linear(block_in_features, block_out_features) for _ in range(num_blocks)])

    def forward(self, x):
        device = x.device
        x = x.chunk(self.num_blocks, dim=-1)

        x = [block(x_i) for block, x_i in zip(self.blocks, x)]
        return torch.cat(x, dim=-1)


class sLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=4 / 3):
        super(sLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.proj_factor = proj_factor

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if proj_factor <= 0:
            raise ValueError("proj_factor must be greater than zero")

        self.layer_norm = nn.LayerNorm(input_size)
        self.causal_conv = CausalConv1D(1, 1, 4)
        self.Wz = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wi = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wf = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wo = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Rz = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ri = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Rf = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ro = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.group_norm = nn.GroupNorm(num_heads, hidden_size)
        self.up_proj_left = nn.Linear(hidden_size, int(hidden_size * proj_factor))
        self.up_proj_right = nn.Linear(hidden_size, int(hidden_size * proj_factor))
        self.down_proj = nn.Linear(int(hidden_size * proj_factor), input_size)

    def forward(self, x, prev_state):
        # print(x.size(-1))
        # print(self.input_size)
        assert x.size(-1) == self.input_size
        h_prev, c_prev, n_prev, m_prev = prev_state

        x_norm = self.layer_norm(x)
        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))

        device = x.device
        h_prev = h_prev.to(device)
        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))
        i_tilde = self.Wi(x_conv) + self.Ri(h_prev)
        f_tilde = self.Wf(x_conv) + self.Rf(h_prev)

        m_prev = m_prev.to(device)
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
        c_prev = c_prev.to(device)
        n_prev = n_prev.to(device)
        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        h_t = o * c_t / n_t

        output = h_t
        output_norm = self.group_norm(output)
        output_left = self.up_proj_left(output_norm)
        output_right = self.up_proj_right(output_norm)
        output_gated = F.gelu(output_right)
        output = output_left * output_gated
        output = self.down_proj(output)

        final_output = output + x
        return final_output, (h_t, c_t, n_t, m_t)


class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers=1, batch_first=False, proj_factor=4 / 3):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.layers = nn.ModuleList(
            [sLSTMBlock(input_size, hidden_size, num_heads, proj_factor) for _ in range(num_layers)])

    def forward(self, x, state=None):
        assert x.ndim == 3

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        if state is None:
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size)
        else:
            state = torch.stack(list(state))
            assert state.ndim == 4 and state.size(0) == self.num_layers

        output = []
        for t in range(seq_len):
            x_t = x[t]
            for layer in range(self.num_layers):
                x_t, state_tuple = self.layers[layer](x_t, tuple(state[layer].clone()))
                state[layer] = torch.stack(list(state_tuple))
            output.append(x_t)

        output = torch.stack(output)

        if self.batch_first:
            output = output.transpose(0, 1)

        state = tuple(state.transpose(0, 1))
        return output, state


class mLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=2):
        super(mLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.proj_factor = proj_factor

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if proj_factor <= 0:
            raise ValueError("proj_factor must be greater than zero")

        self.layer_norm = nn.LayerNorm(input_size)
        self.up_proj_left = nn.Linear(input_size, int(input_size * proj_factor))
        self.up_proj_right = nn.Linear(input_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, input_size)
        self.causal_conv = CausalConv1D(1, 1, 4)
        self.skip_connection = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wq = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wk = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wv = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wi = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wf = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wo = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.group_norm = nn.GroupNorm(num_heads, hidden_size)

    def forward(self, x, prev_state):
        h_prev, c_prev, n_prev, m_prev = prev_state
        assert x.size(-1) == self.input_size

        device = x.device
        x_norm = self.layer_norm(x)
        x_up_left = self.up_proj_left(x_norm)
        x_up_right = self.up_proj_right(x_norm)
        x_conv = F.silu(self.causal_conv(x_up_left.unsqueeze(1)).squeeze(1))
        x_skip = self.skip_connection(x_conv)

        q = self.Wq(x_conv)
        k = self.Wk(x_conv) / (self.head_size ** 0.5)
        v = self.Wv(x_up_left)

        i_tilde = self.Wi(x_conv)
        f_tilde = self.Wf(x_conv)
        o = torch.sigmoid(self.Wo(x_up_left))

        m_prev = m_prev.to(device)
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
        c_prev = c_prev.to(device)
        n_prev = n_prev.to(device)
        c_t = f * c_prev + i * (v * k)
        n_t = f * n_prev + i * k
        h_t = o * (c_t * q) / torch.max(torch.abs(n_t.T @ q), 1)[0]

        output = h_t
        output_norm = self.group_norm(output)
        output = output_norm + x_skip
        output = output * F.silu(x_up_right)
        output = self.down_proj(output)

        final_output = output + x
        return final_output, (h_t, c_t, n_t, m_t)


class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers=1, batch_first=False, proj_factor=2):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.proj_factor = proj_factor
        self.layers = nn.ModuleList(
            [mLSTMBlock(input_size, hidden_size, num_heads, proj_factor) for _ in range(num_layers)])

    def forward(self, x, state=None):
        assert x.ndim == 3

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        if state is None:
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size)
        else:
            state = torch.stack(list(state))
            assert state.ndim == 4 and state.size(0) == self.num_layers

        output = []
        for t in range(seq_len):
            x_t = x[t]
            for layer in range(self.num_layers):
                x_t, state_tuple = self.layers[layer](x_t, tuple(state[layer].clone()))
                state[layer] = torch.stack(list(state_tuple))
            output.append(x_t)

        output = torch.stack(output)

        if self.batch_first:
            output = output.transpose(0, 1)

        state = tuple(state.transpose(0, 1))
        return output, state


class xLSTM(nn.Module):
    def __init__(self, input_size, layers, hidden_size=64, num_heads=1, batch_first=False, proj_factor_slstm=4 / 3,
                 proj_factor_mlstm=2):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = len(layers)
        self.batch_first = batch_first
        self.proj_factor_slstm = proj_factor_slstm
        self.proj_factor_mlstm = proj_factor_mlstm

        self.layers = nn.ModuleList()
        for layer_type in layers:
            if layer_type == 's':
                layer = sLSTMBlock(input_size, hidden_size, num_heads, proj_factor_slstm)
            elif layer_type == 'm':
                layer = mLSTMBlock(input_size, hidden_size, num_heads, proj_factor_mlstm)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            self.layers.append(layer)

        # Ensure output size is (batch_size, seq_len, input_size)
        self.output_linear = nn.Linear(input_size, 1)

    def forward(self, x, state=None):
        assert x.ndim == 3

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        if state is not None:
            state = torch.stack(list(state))
            assert state.ndim == 4
            num_hidden, state_num_layers, state_batch_size, state_input_size = state.size()
            assert num_hidden == 4
            assert state_num_layers == self.num_layers
            assert state_batch_size == batch_size
            assert state_input_size == self.hidden_size
            state = state.transpose(0, 1)
        else:
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size)

        output = []
        for t in range(seq_len):
            x_t = x[t]
            for layer in range(self.num_layers):
                x_t, state_tuple = self.layers[layer](x_t, tuple(state[layer].clone()))
                state[layer] = torch.stack(list(state_tuple))
            output.append(x_t)

        output = torch.stack(output)

        output = self.output_linear(output)  # Project to input_size dimension

        state = tuple(state.transpose(0, 1))
        return output[:, -1, :]

class Mistral(nn.Module):
    def __init__(self, input_dim, seq_len=30, d_model=4, num_heads=1, num_layers=1, output_dim=1):
        super(Mistral, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x, y=None):
        # 添加位置编码
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        if y is not None:  # 训练时提供 target
            y = self.embedding(y) + self.positional_encoding[:, :y.size(1), :]
        else:
            y = torch.zeros_like(x)  # 推理时只用输入预测
        output = self.transformer(x, y)
        output = self.fc(output)
        return output[:, -1, :]


class PatchTST(nn.Module):
    def __init__(self, input_dim, patch_len=1, d_model=4, num_heads=1, num_layers=1, output_dim=1, dropout=0.1):
        super(PatchTST, self).__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim * patch_len, d_model)  # 对每个 patch 进行线性映射
        self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, d_model))  # 可训练的 PE
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,  # 我们只用 encoder
            dropout=dropout,
        )
        self.fc = nn.Linear(d_model, output_dim)  # 输出层

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim, num_channels]
        Returns:
            output: [batch_size, output_dim]
        """
        batch_size, seq_len, input_dim = x.shape

        # 将最后两个维度合并，变成 [batch_size, seq_len, input_dim * num_channels]
        x = x.reshape(batch_size, seq_len, -1)

        # 划分为 patches
        num_patches = seq_len // self.patch_len  # patch 数
        x = x[:, :num_patches * self.patch_len, :]  # 仅保留完整的 patches
        x = x.reshape(batch_size, num_patches, -1)  # [batch_size, num_patches, input_dim * patch_len * num_channels]

        # 映射到 d_model
        x = self.embedding(x)  # [batch_size, num_patches, d_model]

        # 加入位置编码
        x = x + self.positional_encoding[:, :x.size(1), :]  # [batch_size, num_patches, d_model]

        # 调整形状为 [seq_len, batch_size, d_model] 以供 Transformer 使用
        x = x.transpose(0, 1)  # [num_patches, batch_size, d_model]

        # Transformer Encoder
        x = self.transformer(x)  # [num_patches, batch_size, d_model]

        # 平均池化（或取最后一时刻特征）
        x = x.mean(dim=0)  # [batch_size, d_model]

        # 输出
        output = self.fc(x)  # [batch_size, output_dim]
        return output

class StackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size=4, num_layers=2, bias=True):
        super(StackedGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc = nn.Linear(4, 1)

        # 定义多层 GRUCell
        self.gru_cells = nn.ModuleList([
            GRUCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias)
            for i in range(num_layers)
        ])

    def forward(self, x, hidden=None):
        """
        :param x: 输入张量，形状为 (batch_size, seq_len, input_size)
        :param hidden: 初始隐状态，形状为 (num_layers, batch_size, hidden_size)
        :return: 输出张量 (batch_size, seq_len, hidden_size) 和最终隐状态
        """
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            # 初始化隐状态为零
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        outputs = []
        layer_hidden = hidden.clone()

        # 时间步循环
        for t in range(seq_len):
            input_t = x[:, t, :]
            new_layer_hidden = []
            for layer in range(self.num_layers):
                gru_cell = self.gru_cells[layer]
                # 获取上一层的输出作为当前输入
                next_hidden = gru_cell(input_t, layer_hidden[layer])
                new_layer_hidden.append(next_hidden)
                input_t = next_hidden
            layer_hidden = torch.stack(new_layer_hidden, dim=0)  # 更新隐状态

            outputs.append(layer_hidden[-1].clone())  # 保存最顶层的输出

        # 拼接时间步的输出
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        output = self.fc(outputs[:, -1, :])
        return torch.sigmoid(output)

class trainmodel(nn.Module):
    def __init__(self, input_size):
        super(trainmodel, self).__init__()
        # self.gru = PatchTST(input_dim=input_size)
        # self.gru = TRA(input_size=input_size)
        # self.gru = AlphaStock(dim_in=input_size)
        # self.gru = MSU(in_features=input_size)
        # self.gru = NGCU(input_dim=input_size)
        # self.deepgru = DeepDifferentGru(input_size=input_size)
        self.deepgru = StackedGRU(input_size=input_size)

        self.fc1 = nn.Linear(768, 192)
        self.fc2 = nn.Linear(192, 48)
        self.fc3 = nn.Linear(48, 10)
        self.fc5 = nn.Linear(768, 192)
        self.fc6 = nn.Linear(192, 48)
        self.fc7 = nn.Linear(48, 10)
        self.self_attention = nn.MultiheadAttention(40, num_heads=1)
        self.fc4 = nn.Linear(80, 40)
        self.fc = nn.Linear(16, 1)

    def forward(self, basic_data, news_data, policy_data):
        '''
        news start
        '''
        batch_size_basic, num_items_basic, total_basic = basic_data.shape
        batch_size_news, num_items_news, total_news = news_data.shape
        use_cuda = 1
        device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
        fc1 = self.fc1(news_data)
        tan1 = torch.tanh_(fc1)
        fc2 = self.fc2(tan1)
        tan2 = torch.tanh_(fc2)
        fc3 = self.fc3(tan2)
        news_ten = torch.tanh_(fc3)

        fc5 = self.fc5(policy_data)
        tan5 = torch.tanh_(fc5)
        fc6 = self.fc6(tan5)
        tan6 = torch.tanh_(fc6)
        fc7 = self.fc3(tan6)
        policy_ten = torch.tanh_(fc7)

        cat_data = torch.cat((basic_data, news_ten, policy_ten), dim=2)
        forth_data = basic_data[:, :, :4]

        output = self.deepgru(basic_data)
        # output = output.to(device)
        # output = self.fc(output[:,-1,:])
        # output = torch.sigmoid(output)

        return output