from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from weight_init import weight_init_

# def einsum(x, A):
#     return torch.einsum("nctkv,kvw->nctw", [x, A])

def einsum(x : torch.Tensor, A : torch.Tensor):
    """paddle.einsum will be implemented in release/2.2.
    """
    # x = x.transpose((0, 2, 3, 1, 4))
    x = x.transpose(1, 2).transpose(2, 3)
    n, c, t, k, v = x.shape
    k2, v2, w = A.shape
    assert (k == k2 and v == v2), "Args of einsum not match!"
    x = x.reshape((n, c, t, k * v))
    A = A.reshape((k * v, w))
    y = torch.matmul(x.to(torch.float32), A.to(torch.float32))
    return y

def get_hop_distance(num_node, edge, max_hop=1):  # 获取图上两点边距离
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

class Graph():

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == 'fsd10': # from openpose body-25
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                             (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                             (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                             (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                             (21, 14), (19, 14), (20, 19)]
            self.edge = self_link + neighbor_link
            self.center = 8
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                              (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                              (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                              (17, 1), (18, 17), (19, 18), (20, 19), (22, 23),
                              (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'coco_keypoint':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
                              (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12),
                              (11, 13), (12, 14), (13, 15), (14, 16), (11, 12)]
            neighbor_link = [(i, j) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 11
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

class ConvTemporalGraphical(nn.Module):  # 时间维度图卷积

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,  # spatial_kernel_size
                 t_kernel_size=1,  # 时间卷积核大小
                 t_stride=1,  # 时间卷积核步幅
                 t_padding=0,  # 时间卷积核边距填充
                 t_dilation=1):  # 扩展倍数？
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1))

    def forward(self, x : np.ndarray, A):
        assert A.shape[0] == self.kernel_size

        x = self.conv(x)
        n, kc, t, v = x.shape
        x = x.reshape((n, self.kernel_size, kc // self.kernel_size, t, v))
        x = einsum(x, A)

        return x, A

class st_gcn_block(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,  # kernel_size = (temporal_kernel_size, spatial_kernel_size)
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn_block, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        # ?

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,  # 图卷积网络
                                         kernel_size[1])

        self.tcn = nn.Sequential(  # 时间卷积网络
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if not residual:
            self.residual = lambda x : 0  # zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x : x  # iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A

class STGCN(nn.Module):
    """
    ST-GCN model from:
    `"Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition" <https://arxiv.org/abs/1801.07455>`_
    Args:
        in_channels: int, channels of vertex coordinate. 2 for (x,y), 3 for (x,y,z). Default 2.
        edge_importance_weighting: bool, whether to use edge attention. Default True.
        data_bn: bool, whether to use data BatchNorm. Default True.
    """

    def __init__(self,
                 num_classes,
                 in_channels=2,
                 edge_importance_weighting=True,
                 data_bn=True,
                 layout='fsd10',
                 strategy='spatial',
                 device='cpu',
                 **kwargs):
        super(STGCN, self).__init__()
        self.data_bn = data_bn
        self.device = device
        # load graph
        self.graph = Graph(  # 创建图并定义分组策略
            layout=layout,
            strategy=strategy,
        )
        # A = torch.to_tensor(self.graph.A, dtype='float32')
        # self.register_buffer('A', A)  # 取出图中的矩阵，[并将矩阵存储到类中，该矩阵不会参与计算](?)
        A = deepcopy(self.graph.A)
        self.A = A

        # build networks 网络结构
        spatial_kernel_size = A.shape[0]
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.shape[1]) if self.data_bn else (lambda x : x)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.Sequential(
            st_gcn_block(in_channels,
                         64,
                         kernel_size,
                         1,
                         residual=False,
                         **kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, 2, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        )

        # initialize parameters for edge importance weighting 边注意力权重
        if edge_importance_weighting:  # todo: fix parameter problem when enable edge importance weighting
            self.edge_importance = nn.ParameterList([
                nn.parameter(
                    shape=self.A.shape,
                    default_initializer=init.constant(1))  # 默认初始化为常数1
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 池化层

        self.classify = nn.Linear(256, num_classes)

    def init_weights(self):
        """Initiate the parameters.
        """
        for layer in self.get_submodule():
            if isinstance(layer, nn.Conv2d):
                weight_init_(layer, 'Normal', mean=0.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm2d):
                weight_init_(layer, 'Normal', mean=1.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm1d):
                weight_init_(layer, 'Normal', mean=1.0, std=0.02)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x : torch.Tensor):
        # data normalization
        N, C, T, V, M = x.shape
        # 样本数(1), [x, y, 置信度](openpose, 3), 帧(1500), 关节点数量(body_25, 25), 运动员数量(1)
        # x = x.transpose((0, 4, 3, 1, 2))  # N, M, V, C, T
        x = x.transpose(1, 4).transpose(2, 4).transpose(2, 3)
        x = x.reshape((N * M, V * C, T))  # 样本*运动员数量, 帧内关节点矩阵(3, 25), 帧
        if self.data_bn:  # 批量归一化
            x.stop_gradient = False
        x = self.data_bn(x)
        x = x.reshape((N, M, V, C, T))
        # x = x.transpose((0, 1, 3, 4, 2))  # N, M, C, T, V
        x = x.transpose(2, 3).transpose(3, 4)
        x = x.reshape((N * M, C, T, V))

        A = torch.tensor(self.A).to(self.device)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, torch.multiply(A, torch.tensor([importance]).to(self.device)))

        x = self.pool(x)  # NM,C,T,V --> NM,C,1,1
        C = x.shape[1]
        x = torch.reshape(x, (N, M, C)).mean(axis=1)  # N,C,1,1
        x = self.classify(x)
        return x
