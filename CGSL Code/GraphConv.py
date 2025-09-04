import math
import torch
import torch.nn.init as init
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim).float())
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim).float())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)


    def forward(self,  input_feature,adjacency):
        support =torch.sparse.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' ('             + str(self.input_dim) + ' -> '             + str(self.output_dim) + ')'




