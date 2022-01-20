import torch
import torch.nn as nn
from layers import Attn_head, Attn_head_adj
import math
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Linear
from scipy.special import comb
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian

#implementation of intra-hyperedge attention module;   3

class hyperedge_attn(nn.Module):   #intra-hyperedge attention
    def __init__(self, nfeat, nhid, dropout):
        """ Intra-hyperedge attention module. """
        super(hyperedge_attn, self).__init__()
        self.intra_hpyeredge = Attn_head(nfeat, nhid, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(), residual=True)
        self.industry_tensor = Attn_head_adj(nfeat, nhid, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(), residual=True)
        self.dropout = dropout

    def forward(self, x, H, adj, nhid):  # x(32,758,32)H(758,62) and ad(758,758), ad goes directly to industry_tensor via attn_head_ad
        batch = x.size(0)
        stock = x.size(1)
        industry_tensor = self.industry_tensor(x, adj) #(32,758,8), 8 is the out_sz of convolution
        all_hyperedge_fts = torch.randn(0).cuda()
        hyperedge_fts = torch.randn(0).cuda()

        in_edge_index = indu_edge_index(adj) #(2,9826)

        for i in range(H.shape[1]):  #62
            intra_hyperedge_fts = torch.randn(0).cuda()
            node_set = torch.nonzero(H[:, i], as_tuple=False) #758,i

            for j in range(len(node_set)): #6
                node_index = node_set[j]
                intra_hyperedge_fts = torch.cat([intra_hyperedge_fts, x[:, node_index, :]], dim=1)

            after_intra = self.intra_hpyeredge(intra_hyperedge_fts)  #intra_hyperedge (32,6,32), after_intra(32,6,8)
            pooling = torch.nn.MaxPool1d(len(node_set), stride=1)
            e_fts = pooling(after_intra.permute(0, 2, 1))  #  e_fts (32,8,1)
            hyperedge_fts = torch.cat([hyperedge_fts, e_fts.permute(0, 2, 1)], dim=1)

            single_edge = torch.zeros(batch, stock, nhid).cuda()  #（32，758，8）

            for j in range(len(node_set)):#6
                node_index = node_set[j]
                single_edge[:, node_index.squeeze(0), :] = after_intra[:, j, :]   # single edge (32，758，8）  after_intra (32,6,8)

            all_hyperedge_fts = torch.cat([all_hyperedge_fts, single_edge.unsqueeze(0)], dim=0)  #(62,32,758,8)

        return all_hyperedge_fts, hyperedge_fts, industry_tensor   #all_ and hyper all included the fund information
    #all (62,32,758,8)   hyperedge (32,62,8)--after max pooling, industry(32,758,8)

def indu_edge_index(adjacency):
    edge_index = torch.nonzero(adjacency, as_tuple=False).permute(1,0) #9826,2
    return edge_index

def remove_self_loops(edge_index, edge_attr):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`) :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


# def get_laplacian(edge_index, edge_weight: Optional[torch.Tensor] = None,
#                   normalization: Optional[str] = None,
#                   dtype: Optional[int] = None,
#                   num_nodes: Optional[int] = None):
#     r""" Computes the graph Laplacian of the graph given by :obj:`edge_index` and optional :obj:`edge_weight`.
#     normalization (str, optional): The normalization scheme for the graph Laplacian (default: :obj:`None`)
#   """
#     edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
#     if edge_weight is None:
#         edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
#                                  device=edge_index.device)
#
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#
#     row, col = edge_index[0], edge_index[1]
#     deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
#     if normalization is None:
#         # L = D - A.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
#         edge_weight = torch.cat([-edge_weight, deg], dim=0)
#     elif normalization == 'sym':
#         # Compute A_norm = -D^{-1/2} A D^{-1/2}.
#         deg_inv_sqrt = deg.pow_(-0.5)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#         edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
#         # L = I - A_norm.
#         edge_index, tmp = add_self_loops(edge_index, -edge_weight, fill_value=1., num_nodes=num_nodes)
#         assert tmp is not None
#         edge_weight = tmp
#     else:
#         # Compute A_norm = -D^{-1} A.
#         deg_inv = 1.0 / deg
#         deg_inv.masked_fill_(deg_inv == float('inf'), 0)
#         edge_weight = deg_inv[row] * edge_weight
#
#         # L = I - A_norm.
#         edge_index, tmp = add_self_loops(edge_index, -edge_weight, fill_value=1., num_nodes=num_nodes)
#         assert tmp is not None
#         edge_weight = tmp
#
#     return edge_index, edge_weight

class Bern_prop(MessagePassing): #has to use MessagePassing rather than nn.Module
    def __init__(self, K, bias=True, **kwargs): #K propagation steps default 10
        super(Bern_prop, self).__init__(**kwargs) #edge_index is necessary
        self.K = K
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)
        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype, num_nodes=758) #x.size(self.node_dim)
        # 2I-L
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=758)  #x.size(self.node_dim)

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]
        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)

            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class BernNet(torch.nn.Module):
    def __init__(self, features, nhid, n_classes, K, dprate, dropout): #dprate dropout for propagation layer, default 0.5
        super(BernNet, self).__init__()
        self.lin1 = Linear(features, nhid)
        self.lin2 = Linear(nhid, features) #change n_classes to nfeat as want the same dim        #self.m = torch.nn.BatchNorm1d(n_classes)
        self.prop1 = Bern_prop(K)
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):  #change data to x, edge_index
        #x, edge_index = data.x, data.edge_index  #x train_mast   #x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.dropout(x, p=self.dropout)
        x = F.relu(self.lin1(x))   #x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.dropout(x, p=self.dropout)
        x = self.lin2(x)  # x= self.m(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x             #return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate)   ##x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x #return F.log_softmax(x, dim=1)






class hyperedge(nn.Module):   #intra-hyperedge attention
    def __init__(self, nfeat, nhid, dropout):
        """ Intra-hyperedge attention module. """
        super(hyperedge, self).__init__() #not use attention or may need to use attention
        #self.intra_hpyeredge = Attn_head(nfeat, nhid, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(), residual=True)
        #self.industry_tensor = Attn_head_adj(nfeat, nhid, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(), residual=True)
        self.dropout = dropout


    def forward(self, x, H, adj, nhid):  # x(32,758,32) H(758,62) and ad(758,758), ad goes directly to industry_tensor via attn_head_ad
        batch = x.size(0)
        stock = x.size(1)
        industry_tensor = self.industry_tensor(x, adj) #(32,758,8), 8 is the out_sz of convolution
        #industry_tensor =   #use bernet
        all_hyperedge_fts = torch.randn(0).cuda()
        hyperedge_fts = torch.randn(0).cuda()

        edge_index = indu_edge_index(adj)

        for i in range(H.shape[1]):  #62
            intra_hyperedge_fts = torch.randn(0).cuda()
            node_set = torch.nonzero(H[:, i], as_tuple=False) #758,i

            for j in range(len(node_set)): #6
                node_index = node_set[j]
                intra_hyperedge_fts = torch.cat([intra_hyperedge_fts, x[:, node_index, :]], dim=1)

            after_intra = self.intra_hpyeredge(intra_hyperedge_fts)  #intra_hyperedge (32,6,32), after_intra(32,6,8)
            pooling = torch.nn.MaxPool1d(len(node_set), stride=1)
            e_fts = pooling(after_intra.permute(0, 2, 1))  #  e_fts (32,8,1)
            hyperedge_fts = torch.cat([hyperedge_fts, e_fts.permute(0, 2, 1)], dim=1)

            single_edge = torch.zeros(batch, stock, nhid).cuda()  #（32，758，8）

            for j in range(len(node_set)):#6
                node_index = node_set[j]
                single_edge[:, node_index.squeeze(0), :] = after_intra[:, j, :]   # single edge (32，758，8）  after_intra (32,6,8)

            all_hyperedge_fts = torch.cat([all_hyperedge_fts, single_edge.unsqueeze(0)], dim=0)  #(62,32,758,8)

        return all_hyperedge_fts, hyperedge_fts, industry_tensor

# def indu_edge_index(adj):
#     for i in range(adj.shape[0]):
#         node_set = torch.nonzero(adj[i, :], as_tuple=False)
#         edge_index = torch.tensor([[node_set],[i]])
#     return edge_index