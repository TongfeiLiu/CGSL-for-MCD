import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,  GINConv, SAGEConv,SGConv,GATConv
import torch.nn.functional as F
from GraphConv import GraphConvolution



# 定义图卷积层
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type):
        super(GraphConv, self).__init__()
        # 根据conv_type参数选择不同类型的图卷积层
        if conv_type == 'GCN':
            self.conv = GCNConv(in_channels, out_channels)
        elif conv_type == 'GAT':
            self.conv = GATConv(in_channels, out_channels,dropout=0.1)
        elif conv_type == 'GIN':
            self.conv = GINConv(nn.Linear(in_channels, out_channels))
        elif conv_type == 'ST':
            self.conv = SAGEConv(in_channels, out_channels,normalize=True)
        elif conv_type == 'SGC':
            self.conv = SGConv(in_channels,out_channels)
        elif conv_type == 'Con':
            self.conv =GraphConvolution(in_channels,out_channels)
        else:
            raise ValueError('Invalid conv_type')


    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

class ClampedLeakyReLU(nn.Module):
    def __init__(self, lower_bound=-3, upper_bound=1):
        super(ClampedLeakyReLU, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, x):
        return torch.clamp(F.leaky_relu(x), min=self.lower_bound, max=self.upper_bound)



# 定义编码器
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels, conv_type='Con')
        self.conv2_mean = GraphConv(hidden_channels, out_channels, conv_type='Con')
        self.conv2_logstd = GraphConv(hidden_channels, out_channels, conv_type='Con')
        self.clampedLeakyReLU = ClampedLeakyReLU()


    def forward(self, x, edge_index):
        x_emp = self.conv1(x, edge_index)
        x =  self.clampedLeakyReLU(x_emp)

        mean_emp = self.conv2_mean(x, edge_index)
        mean = self.clampedLeakyReLU(mean_emp)

        logstd_emp = self.conv2_logstd(x, edge_index)
        logstd = self.clampedLeakyReLU(logstd_emp)

        return x, mean, logstd, x_emp, mean_emp,  logstd_emp

class Decoder_a(nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels):
        super(Decoder_a, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels, conv_type='Con')
        self.conv2 = GraphConv(hidden_channels, out_channels, conv_type='Con')


    def forward(self, z,edge_index):
        z_emp = self.conv1(z, edge_index)
        z_have = (torch.tanh(z_emp))
        z_2emp = self.conv2(z_have, edge_index)
        z = torch.tanh(z_2emp)
        return z

class Decoder_b(nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, dropout_rate=0.3):
        super(Decoder_b, self).__init__()
        self.conv1 = GraphConv(in_channels,  hidden_channels, conv_type='Con')
        self.conv2 = GraphConv(hidden_channels, out_channels, conv_type='Con')


    def forward(self,w,edge_index):
        w_emp = self.conv1(w, edge_index)
        w_have = (torch.tanh(w_emp))
        w_2emp = self.conv2(w_have, edge_index)
        w = torch.tanh(w_2emp)
        return w









