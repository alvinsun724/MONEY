import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Parameter
import math
np.set_printoptions(threshold=np.inf) #no ...for array
#implementation of attention layer;    2



class Attn_head_adj(nn.Module):    #for hyperedge_attn
    def __init__(self,
                 in_channel,
                 out_sz,
                 in_drop=0.0,
                 coef_drop=0.0,
                 activation=None,
                 residual=False):
        super(Attn_head_adj, self).__init__()

        self.in_channel = in_channel
        self.out_sz = out_sz
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.activation = activation
        self.residual = residual

        self.conv1 = nn.Conv1d(self.in_channel, self.out_sz, 1)
        self.conv2_1 = nn.Conv1d(self.out_sz, 1, 1)
        self.conv2_2 = nn.Conv1d(self.out_sz, 1, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.in_dropout = nn.Dropout()
        self.coef_dropout = nn.Dropout()
        self.res_conv = nn.Conv1d(self.in_channel, self.out_sz, 1)

    def forward(self, x, adj):
        seq = x.permute(0, 2, 1)   #seq 32 32 758
        seq_fts = self.conv1(seq)   #32,8,758, 8 is the out_sz
        f_1 = self.conv2_1(seq_fts)
        f_2 = self.conv2_2(seq_fts)
        logits = f_1 + torch.transpose(f_2, 2, 1)
        logits = self.leakyrelu(logits)

        zero_vec = -9e15 * torch.ones_like(logits)  #(32,758,758)
        attention = torch.where(adj > 0, logits, zero_vec) #if adj then return value in logits

        coefs = self.softmax(attention)

        if self.coef_drop != 0.0:
            coefs = self.coef_dropout(coefs)
        if self.in_dropout != 0.0:
            seq_fts = self.in_dropout(seq_fts)

        ret = torch.matmul(coefs, torch.transpose(seq_fts, 2, 1))  #coefs(32,758,758) seq_fts(32,8,758) , ret (32,758,8)

        if self.residual:
            if seq.shape[1] != ret.shape[2]:
                ret = ret + self.res_conv(seq).permute(0, 2, 1)
            else:
                ret = ret + seq.permute(0, 2, 1)

        return self.activation(ret)  #32,758,8   8 is the out_sz of convolution

class Attn_head(nn.Module):    #for hyperedge_attn
    def __init__(self,
                 in_channel,
                 out_sz,     #out_sz = 8
                 in_drop=0.0,
                 coef_drop=0.0,
                 activation=None,
                 residual=False):
        super(Attn_head, self).__init__()

        self.in_channel = in_channel
        self.out_sz = out_sz
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.activation = activation
        self.residual = residual

        self.conv1 = nn.Conv1d(self.in_channel, self.out_sz, 1)
        self.conv2_1 = nn.Conv1d(self.out_sz, 1, 1)
        self.conv2_2 = nn.Conv1d(self.out_sz, 1, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.in_dropout = nn.Dropout()
        self.coef_dropout = nn.Dropout()
        self.res_conv = nn.Conv1d(self.in_channel, self.out_sz, 1)

    def forward(self, x):  #(32,6,32)
        seq = x.permute(0, 2, 1)   #(32,32,6)
        seq_fts = self.conv1(seq)  #32,8,6
        f_1 = self.conv2_1(seq_fts)  #32,1,6
        f_2 = self.conv2_2(seq_fts)   #32,1,6
        logits = f_1 + torch.transpose(f_2, 2, 1)   #32,6,6
        logits = self.leakyrelu(logits)
        coefs = self.softmax(logits)   #32,6,6

        if self.coef_drop != 0.0:
            coefs = self.coef_dropout(coefs)
        if self.in_dropout != 0.0:
            seq_fts = self.in_dropout(seq_fts)
        ret = torch.matmul(coefs, torch.transpose(seq_fts, 2, 1))   #32,6,8

        if self.residual:   #seq(32,32,6)
            if seq.shape[1] != ret.shape[2]:
                ret = ret + self.res_conv(seq).permute(0, 2, 1)
            else:
                ret = ret + seq.permute(0, 2, 1)

        return self.activation(ret)    #ret (32,6,8)

class edge_attn(nn.Module):
    def __init__(self,
                 in_channel,
                 out_sz):
        super(edge_attn, self).__init__()
        self.in_channel = in_channel #8
        self.out_sz = out_sz   #16

        self.conv1 = nn.Conv1d(self.in_channel, self.out_sz, 1) #8，16，1, this 1 is for 32
        self.conv2_1 = nn.Conv1d(self.out_sz, 1, 1) #16,1,1
        self.conv2_2 = nn.Conv1d(self.out_sz, 1, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        seq_fts = self.conv1(x.permute(0, 2, 1))   #x(32,2,8)-> permute (32,8,2)  seq(32,16,2)
        f_1 = self.conv2_1(seq_fts)  #32,1,2
        f_2 = self.conv2_2(seq_fts)  #32,1,2
       # y5 = torch.transpose(f_2, 2, 1)  #(32,2,1)
        logits = f_1 + torch.transpose(f_2, 2, 1)  #  (32,1,2) +    (32,2,1)  ->    (32,2,2)
        logits = self.leakyrelu(logits)

        coefs = self.softmax(logits)   #logits(32,2,2)
        coefs = torch.sum(coefs, dim=-1, out=None) #(32,2)
        coefs = self.softmax(coefs)
        coefs = coefs.unsqueeze(1)
        return coefs   #(32,1,2)



class EdgeConv(nn.Module):
    def __init__(self,in_channel, out_sz):
        super(EdgeConv, self).__init__()
        self.in_channel = in_channel  # 8
        self.out_sz = out_sz # 8
        self.mlp = MLP(16*in_channel, out_sz)
        self.conv1 = nn.Conv1d(self.in_channel, self.out_sz, 1)
        self.conv2_1 = nn.Conv1d(self.out_sz, 1, 1)  # 8
        self.conv2_2 = nn.Conv1d(self.out_sz, 8, 1)
        self.weight = Parameter(torch.FloatTensor(in_channel, out_sz)) #change from 2,1 to 1 1
        self.weight1 = Parameter(torch.FloatTensor(2, 1))
        self.reset_parameters

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x): # x(32,2,8)
        #seq_fts = x.permute(0, 2, 1)  #  seq(32,8,2)
        x02 = []
        for i in range(x.shape[0]):
            x00 = x[i,:,:]
            x01 = torch.mm(x00, self.weight) #mat1 and mat2 shapes cannot be multiplied (8x2 and 8x8)
            #x_1 = x_ * self.weight
            x02.append(x01)
        support = torch.stack(x02, dim=0) #32,2,8
        batch = x.shape[0]
        d_8 = x.shape[2]
        edge_features = support[:,1,:].unsqueeze(1)

        #edge_features = support.view(batch,d_8,-1)

        #y2 = torch.mm(edge_features, self.weight1)

        # y = self.conv2_2(edge_features)# (32,16,1) can do with conv2(16,8,1)  as (32, 8, 1)
        # #y = self.conv2_2(edge_features)   #if cat once: conv2_2 (8,1,2) then (32,1,3); conv2_2 (8,1,1) then (32,1,4); conv2_2 (8,1,4) then (32,1,1)
        # y1 = y.permute(0,2,1)
        # split_list =[1,1]
        #
        # y_s1, y_s2 = y1.split(split_list,dim=1)
        # y2 = y_s1 + y_s2
        return edge_features


# class EdgeConv(nn.Module):
#     def __init__(self,in_channel, out_sz):
#         super(EdgeConv, self).__init__()
#         self.in_channel = in_channel  # 8
#         self.out_sz = out_sz # 8
#         self.mlp = MLP(16*in_channel, out_sz)
#         self.conv1 = nn.Conv1d(self.in_channel, self.out_sz, 1)
#         self.conv2_1 = nn.Conv1d(self.out_sz, 1, 1)  # 8
#         self.conv2_2 = nn.Conv1d(2*self.out_sz, 8, 1)
#         self.weight = Parameter(torch.FloatTensor(in_channel, out_sz)) #change from 2,1 to 1 1
#         self.reset_parameters
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#
#     def forward(self, x, z): # x(32,2,8)
#         #seq_fts = x.permute(0, 2, 1)  #  seq(32,8,2)
#         #seq_z = z.permute(0, 2, 1)
#         #seq_fts = self.conv1(x.permute(0, 2, 1))    #32,8,2
#         #seq_z = self.conv1(z.permute(0, 2, 1))
#         x02 = []
#         x04 = []
#         c = x+z
#         # for i in range(x.shape[0]):
#         #     x00 = z[i,:,:]
#         #     x_ = x[i,:,:]
#         #     x01 = torch.mm(x00, self.weight) #mat1 and mat2 shapes cannot be multiplied (8x2 and 8x8)
#         #     x_1 = torch.mm(x_, self.weight)
#         #     #x_1 = x_ * self.weight
#         #     x02.append(x01)
#         #     x04.append(x_1)
#         # support = torch.stack(x02, dim=0) #32,2,8
#         # su = torch.stack(x04,dim=0)
#         # batch = x.shape[0]
#         # d_16 = x.shape[2]+ x.shape[2]
#         #f_1 = self.conv1(seq_fts)  # 32,1,2,  now 32,8,2
#         #f_2 = self.conv1(seq_fts) #32,1,2    now 32,8,2
#         #logits = f_1 - torch.transpose(f_2, 2, 1)  # (32,2,2)
#         #logits = torch.transpose(seq_fts, 2, 1)  #(32,2,8)
#
#         # edge_features = torch.cat([seq_fts, seq_z-seq_fts], dim =1)  #(32,16,2)
#         # #y = self.conv2_1(edge_features)
#         # y = self.conv2_2(edge_features)  #32,1,2 if conv2_2 (16,1,1), if conv2_2 (16,4,1) y (32,4,2), if conv2_2 (16,1,4) then kernel size cannot greater than actual input size
#         # #y = self.mlp(edge_features)   #should be (32,1,8) now, 32,2,8
#         # #y1 = y[:,1,:].unsqueeze(1) #y[:,1,:] is (32,8) no dim at 1
#         #return y1
#         #edge_features = torch.cat([seq_fts, seq_z-seq_fts], dim =2)  #32,8,4
#         #edge_features = seq_fts + support.permute(0,2,1) #cannot directly add
#
#         for i in range(x.shape[0]):
#             x00 = c[i,:,:]
#             x01 = torch.mm(x00, self.weight) #mat1 and mat2 shapes cannot be multiplied (8x2 and 8x8)
#             #x_1 = x_ * self.weight
#             x02.append(x01)
#         support = torch.stack(x02, dim=0) #32,2,8
#         batch = x.shape[0]
#         d_16 = x.shape[2]+ x.shape[2]
#
#         edge_features = support.view(batch,d_16,-1)
#         y = self.conv2_2(edge_features)# (32,16,1) can do with conv2(16,8,1)  as (32, 8, 1)
#         #y = self.conv2_2(edge_features)   #if cat once: conv2_2 (8,1,2) then (32,1,3); conv2_2 (8,1,1) then (32,1,4); conv2_2 (8,1,4) then (32,1,1)
#         y1 = y.permute(0,2,1)
#         return y1


class MLP(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = Linear(input_dim, out_dim)
        self.fc2 = Linear(out_dim, out_dim)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x