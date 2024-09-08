import dgl
import torch
import torch.nn as nn
import dgl.nn.functional as F
import dgl.nn as dglnn

# 定义异构图节点分类模型
class HeteroNodeClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(HeteroNodeClassifier, self).__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            ('stock', 'stock_financial', 'financial'): dglnn.GraphConv(5, hidden_size),
            ('stock', 'stock_macro', 'macro'): dglnn.GraphConv(5, hidden_size),
            ('stock', 'stock_news', 'news'): dglnn.GraphConv(5, hidden_size),
            ('stock', 'stock_policy', 'policy'): dglnn.GraphConv(5, hidden_size),
            ('financial', 'financial_macro', 'macro'): dglnn.GraphConv(16, hidden_size),
            ('financial', 'financial_stock', 'stock'): dglnn.GraphConv(16, hidden_size),
            ('financial', 'financial_news', 'news'): dglnn.GraphConv(16, hidden_size),
            ('financial', 'financial_policy', 'policy'): dglnn.GraphConv(16, hidden_size),
            ('macro', 'macro_stock', 'stock'): dglnn.GraphConv(21, hidden_size),
            ('macro', 'macro_financial', 'financial'): dglnn.GraphConv(21, hidden_size),
            ('macro', 'macro_news', 'news'): dglnn.GraphConv(21, hidden_size),
            ('macro', 'macro_policy', 'policy'): dglnn.GraphConv(21, hidden_size),
            ('news', 'news_stock', 'stock'): dglnn.GraphConv(768, hidden_size),
            ('news', 'news_financial', 'financial'): dglnn.GraphConv(768, hidden_size),
            ('news', 'news_macro', 'macro'): dglnn.GraphConv(768, hidden_size),
            ('news', 'news_policy', 'policy'): dglnn.GraphConv(768, hidden_size),
            ('policy', 'policy_stock', 'stock'): dglnn.GraphConv(768, hidden_size),
            ('policy', 'policy_macro', 'macro'): dglnn.GraphConv(768, hidden_size),
            ('policy', 'policy_news', 'news'): dglnn.GraphConv(768, hidden_size),
            ('policy', 'policy_financial', 'financial'): dglnn.GraphConv(768, hidden_size)
        }, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            ('stock', 'stock_financial', 'financial'): dglnn.GraphConv(hidden_size, hidden_size),
            ('stock', 'stock_macro', 'macro'): dglnn.GraphConv(hidden_size, hidden_size),
            ('stock', 'stock_news', 'news'): dglnn.GraphConv(hidden_size, hidden_size),
            ('stock', 'stock_policy', 'policy'): dglnn.GraphConv(hidden_size, hidden_size),
            ('financial', 'financial_macro', 'macro'): dglnn.GraphConv(hidden_size, hidden_size),
            ('financial', 'financial_stock', 'stock'): dglnn.GraphConv(hidden_size, hidden_size),
            ('financial', 'financial_news', 'news'): dglnn.GraphConv(hidden_size, hidden_size),
            ('financial', 'financial_policy', 'policy'): dglnn.GraphConv(hidden_size, hidden_size),
            ('macro', 'macro_stock', 'stock'): dglnn.GraphConv(hidden_size, hidden_size),
            ('macro', 'macro_financial', 'financial'): dglnn.GraphConv(hidden_size, hidden_size),
            ('macro', 'macro_news', 'news'): dglnn.GraphConv(hidden_size, hidden_size),
            ('macro', 'macro_policy', 'policy'): dglnn.GraphConv(hidden_size, hidden_size),
            ('news', 'news_stock', 'stock'): dglnn.GraphConv(hidden_size, hidden_size),
            ('news', 'news_financial', 'financial'): dglnn.GraphConv(hidden_size, hidden_size),
            ('news', 'news_macro', 'macro'): dglnn.GraphConv(hidden_size, hidden_size),
            ('news', 'news_policy', 'policy'): dglnn.GraphConv(hidden_size, hidden_size),
            ('policy', 'policy_stock', 'stock'): dglnn.GraphConv(hidden_size, hidden_size),
            ('policy', 'policy_macro', 'macro'): dglnn.GraphConv(hidden_size, hidden_size),
            ('policy', 'policy_news', 'news'): dglnn.GraphConv(hidden_size, hidden_size),
            ('policy', 'policy_financial', 'financial'): dglnn.GraphConv(hidden_size, hidden_size)
        }, aggregate='sum')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, G, node_feats):
        h = self.conv1(G, node_feats)
        h = self.conv2(G, h)
        with G.local_scope():
            # 合并所有节点特征到一个统一的节点类型
            G.nodes['stock'].data['h'] = h['stock']  # 假设用户节点的特征是我们要用到的
            for ntype in G.ntypes:
                if ntype != 'stock':
                    G.nodes['stock'].data['h'] += h[ntype]
            # 对合并后的节点类型进行平均操作
            G.nodes['stock'].data['h'] /= len(G.ntypes)
            dgl_m = dgl.mean_nodes(G, 'h', ntype='stock')
            output = self.fc(dgl_m)
            return G, output


class AttentionGraph(nn.Module):
    def __init__(self):
        super(AttentionGraph, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(16, 10)
        self.fc3 = nn.Linear(21, 10)
        self.fc4 = nn.Linear(768, 10)
        self.fc5 = nn.Linear(768, 10)
        self.count = nn.Linear(10, 1)
        self.pre = nn.Linear(10, 1)

    def forward(self, stock, financial, macro, news, policy):
        stock_df = self.fc1(stock)
        financial_df = self.fc2(financial)
        macro_df = self.fc3(macro)
        news_df = self.fc4(news)
        policy_df = self.fc5(policy)
        total_df = torch.cat((stock_df, financial_df, macro_df, news_df, policy_df), dim=1)
        attention = torch.sigmoid_(self.count(total_df))
        weighted_sum = torch.sum(total_df * attention, dim=1)
        predict = self.pre(weighted_sum)

        return weighted_sum, predict