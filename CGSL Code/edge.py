import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel


def gaussian_kernel_distance(vector, band_width):
    euc_dis = pairwise_distances(vector)#计算输入向量集合 中所有向量之间的欧氏距离。也就是每个节点里面的得到的 euc_dis 是一个距离矩阵，其中包含了每对向量之间的欧氏距离。，pairwise_distances 函数是 scikit-learn 库提供的一个用于计算一组向量之间距离的工具函数。这个函数的目的是计算给定集合中所有向量之间的距离，并返回一个距离矩阵。
    gaus_dis = np.exp(- euc_dis * euc_dis / (band_width * band_width))#通过对欧氏距离矩阵进行高斯核计算，得到高斯相似性矩阵
    return gaus_dis  #高斯相似性是一个矩阵


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix.归一化后的邻接矩阵。这个归一化过程通常用于在图神经网络中提高模型的稳定性和性能。"""
    # adj = np.coo_matrix(adj) np.coo_max
    rowsum = np.array(adj.sum(1))  #二维数组每一行求和 D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() #取倒数平方根，展成一维数组 # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.#把求和里面的无穷替换成0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)  # D^-0.5#创建了一个对角矩阵 d_mat_inv_sqrt，其中对角线上的元素是数组 d_inv_sqrt 中的元素。
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt) #谱图理论或图拉普拉斯矩阵的归一化 # D^-0.5AD^0.5


def construct_affinity_matrix(data, objects, band_width):
    am_set = []
    obj_nums = np.max(objects)
    for i in range(0, obj_nums + 1):
        sub_object = data[objects == i]#把超像素对应到原始图像中，这样做的目的是将超像素映射回原始图像中。
        adj_mat = gaussian_kernel_distance(sub_object, band_width=band_width)
        norm_adj_mat = normalize_adj(adj_mat)#谱图理论或图拉普拉斯矩阵的归一化
        am_set.append([adj_mat, norm_adj_mat])
    return am_set #函数返回一个包含每个子集亲和矩阵的集合am_set



#
#

