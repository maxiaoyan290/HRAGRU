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


# 定义HAN模型
class HANModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_types, metadata):
        super(HANModel, self).__init__()
        self.metadata = metadata
        self.conv1 = HANConv(in_channels, hidden_channels, heads=1, metadata=self.metadata)
        self.conv2 = HANConv(hidden_channels, out_channels, heads=1, metadata=self.metadata)

    def forward(self, data):
        x_dict = {}
        # 构建节点特征字典，并包含完整的节点键
        for key in data.keys():
            if '_x_' not in key and key != 'edge_index_dict':
                x_dict[key] = data[key]
        edge_dict = data.edge_index_dict
        # 将 x_dict 中的张量移到 CUDA 上
        x_dict_cuda = {key: tensor.to('cuda') for key, tensor in x_dict.items()}

        # 将 edge_dict 中的张量移到 CUDA 上
        edge_dict_cuda = {key: tensor.to('cuda') for key, tensor in edge_dict.items()}
        x = self.conv1(x_dict, edge_dict)
        out_dict = self.conv2(x, edge_dict)

        sum_tensor = torch.zeros(1, 10).to('cuda')
        for tensor in out_dict.values():
            if tensor.shape[0] != 1:
                tensor = tensor[0, :].unsqueeze(0)
            #print(tensor)
            sum_tensor += tensor
        num_tensors = len(out_dict)
        average_tensor = sum_tensor / num_tensors

        return average_tensor


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
        i_i = torch.tanh(i_i)
        h_i = torch.tanh(h_i)
        attention_tmp = torch.cat((i_i, h_i), dim=-1)
        #         print(i_i.size())
        #         print(attention_tmp.size())

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(self.attention(attention_tmp))
        newgate = torch.tanh(i_n + (resetgate * h_n))

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


class DeepGru(nn.Module):
    def __init__(self, input_size):
        super(DeepGru, self).__init__()
        self.gru_1 = GRUModel(input_size, 1024, 1)
        self.gru_2 = GRUModel(1024, 512, 1)
        self.gru_3 = GRUModel(512, 256, 1)
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
        self.graphnn = HANModel(in_channels= in_channels,hidden_channels= hidden_channels,out_channels= out_channels,num_types= num_types,metadata= metadata)
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
    def __init__(self, input_dim, output_dim, num_layers=6, num_heads=8, hidden_dim=512, dropout=0.1):
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
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

        out = self.fc(out[:, -1, :])  # Get output from last time step

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
        out_3 = self.dropout(out_3).to('cuda')
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out = self.linear_3(out_5)
        return out

class testmodelnew(nn.Module):
    def __init__(self, input_size):
        super(testmodelnew, self).__init__()
        self.deepoldgru = DeepOldGru(input_size=40)
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
        output = self.deepoldgru(total_data_0)

        return output
