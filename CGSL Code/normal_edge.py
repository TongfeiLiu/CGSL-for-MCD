import torch
def feature_normalization(adj_matrix):
    # 计算每个节点的度数
    degree = adj_matrix.sum(1)
    # 计算每个节点的度数的平方根的倒数
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0  # 处理无穷值
    # 构建对角矩阵，其中对角线上的元素是节点度数的平方根的倒数
    D_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
    # 将邻接矩阵和特征矩阵进行特征值归一化
    normalized_adj_matrix = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    return normalized_adj_matrix



