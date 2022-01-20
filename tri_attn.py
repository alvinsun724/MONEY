"""Hypergraph Tri-attention Network."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperedge_attn import hyperedge_attn
from layers import edge_attn, EdgeConv
from torch.nn import Parameter
import math
#inter-hyperedge attention module and inter-hypergraph attention module      4

class HGN_AD(nn.Module):
    """Tri-attention Modules."""
    def __init__(self, nfeat, nhid, dropout,hinge=0):    #nfeat = rnn_unit ; add hinge
        super(HGN_AD, self).__init__()
        self.hyperedge = hyperedge_attn(nfeat, nhid, dropout) #intra-hyperedge attention
        self.attn = edge_attn(nhid, nhid*2)    #from layers   inter-hyperedge attention
        self.conv = EdgeConv(nhid, nhid)
        self.dropout = dropout
        if hinge == 1:
            self.hinge = True
        if hinge == 0:
            self.hinge = False
    # def adv_part(self, adv_inputs, hidden):
    #     print('adversial part')
    #     self.fc_W = Parameter(torch.FloatTensor(hidden*2, 1))
    #     self.fc_b = Parameter(torch.FloatTensor(1))
    #     if self.hinge:
    #         out = torch.matmul(adv_inputs, self.fc_W) #use matmul as x is 3 dim, weight is 2 dim
    #         out = out + self.bias
    #     else:
    #         out = torch.matmul(self.fea_con, self.fc_W)  # use matmul as x is 3 dim, weight is 2 dim
    #         out = torch.nn.Sigmoid(out + self.bias)
    #     return out

    # def construct_graph(self, x, hidden):
    #     self.av_w = Parameter(torch.FloatTensor(hidden, hidden))
    #     self.av_b = Parameter(torch.FloatTensor(hidden))
    #     self.av_u = Parameter(torch.FloatTensor(hidden))
    #     stdv = 1. / math.sqrt(self.av_w.size(1))
    #     self.av_w.data.uniform_(-stdv, stdv)
    #     self.av_b.data.uniform_(-stdv, stdv)
    #     self.av_.data.uniform_(-stdv, stdv)
    #
    #     self.a_laten = F.tanh(torch.tensordot(x, self.av_w, 1) + self.av_b)  #change self.output to x
    #     self.a_scores = torch.tensordot(self.a_laten, self.av_u, 1)
    #     self.a_alphas = F.softmax(self.a_scores)
    #     self.a_con = torch.sum(x * self.a_alphas.unsqueeze(-1), 1)  #change self.output to x
    #     self.fea_con = torch.cat([x[:,-1,:], self.a_con],1) #maybe not :,-1,: depend on the output, change self.output to x
    #     return self.fea_con

    def forward(self, x, H, adj, nhid):    #x(32,758,32) is enc_ooutput  , add hidden --rnn 32
        x = F.dropout(x, self.dropout, training=self.training)   #x(32,758,32)  H(758,62) nhid 8 adj(758,758)
        # x = self.construct_graph(x, hidden)
        # self.pred = self.adv_part(x, hidden)
        # if self.hinge:
        #     self.loss = nn.HingeEmbeddingLoss(self.gt, self.pred)
        # else:
        #     self.loss = nn.NLLLoss(self.gt, self.pred)
        # self.delta_adv = torch.autograd.grad(self.loss, [self.fea_con])[0]
        # self.delta_adv.detach()
        # self.delta_adv = F.normalize(self.delta_adv, dim=1)
        # self.adv_pv_var = x + 1e-2 * self.delta_adv
        # self.adv_pred = self.adv_part(self.adv_pv_var, hidden)
        # if self.hinge:
        #     self.adv_loss = nn.HingeEmbeddingLoss(self.gt, self.adv_pred)
        # else:
        #     self.adv_loss = nn.NLLLoss(self.gt, self.adv_pred)

        all_hyperedge_tensor, hyperedge_tensor, industry_tensor = self.hyperedge(x, H, adj, nhid) #all(62,32,758,8), hyp(32,62,8) indus_ten(32,758,8)
        #all_hyperedge_fts, hyperedge_fts, industry_tensor  # all_ and hyper all included the fund information
        # all (62,32,758,8)   hyperedge (32,62,8)--after max pooling, industry(32,758,8)
        all_hyperedge_tensor = F.elu(all_hyperedge_tensor)   #(62,32,758,8)
        hyperedge_tensor = F.elu(hyperedge_tensor)
        final_tensor = torch.randn(0).cuda()   #start from 0

        for i in range(x.shape[1]):#758
            final_fts_i = torch.randn(0).cuda()
            hyperedge_fts_i = torch.randn(0).cuda()
            if torch.sum(H, 1)[i] > 1:
                # vertex degree > 1   #hyperedge(2,1), 2 is two nonzero, 1 is input dim ([[0],[1]]) are the index of nonzero of i=0
                hyperedge = torch.nonzero(H[i], as_tuple=False)  #check this shape, view, nonzero return
                for j in range(len(hyperedge)): #i=0, len=2
                    hyperedge_num = hyperedge[j] #[0]
                    #y0 = all_hyperedge_tensor[hyperedge_num, :, i, :] #i,j=0 (1,32,8), no consider of 758 dim
                    #y1 = all_hyperedge_tensor[hyperedge_num, :, i, :].permute(1, 0, 2) #(32,1,8)
                    final_fts_i = torch.cat([final_fts_i, all_hyperedge_tensor[hyperedge_num, :, i, :].permute(1, 0, 2)], dim=1)#all_hyperedge(62,32,758,8)
                    #final_fts_i (32,1,8), cat on 1 dim which is fund dim, after cat on dim 1, become (32,2,8)
                    #y3 = hyperedge_tensor[:, hyperedge_num, :]  #(32,1,8) also start from fund
                    hyperedge_fts_i = torch.cat([hyperedge_fts_i, hyperedge_tensor[:, hyperedge_num, :]], dim=1) #hyperedge_tensor (32,62,8) check the view

                ## coefs = self.attn(hyperedge_fts_i) #inter-hyperedge attention, i=0,j=1 coefs (32,1,2)
                ## final_fts_i = torch.matmul(coefs, final_fts_i) #i=0, final_fts_i (32,2,8)->(32,1,8)
                y8 = self.conv(final_fts_i)  #add final, delete hyperedge_fts_i
                indus_fund = torch.cat([industry_tensor[:, i, :].unsqueeze(1), y8], dim=1) #indus_ten(32,758,8)   -> indus_fund(32,2,8)
                ###indus_fund = industry_tensor[:, i, :].unsqueeze(1) + y8
                ## coefs = self.attn(indus_fund) #inter-hyperedge attention or inter-hypergraph attention? coefs(32,1,2)
                ## final_indus_fund = torch.matmul(coefs, indus_fund) #final: (32,1,8)
                y9 = self.conv(indus_fund)  #add final   #####can check more 1230 important, with two indus_fund  add final, delete hyperedge_fts_i
                final_tensor = torch.cat([final_tensor, y9], dim=1)  #(32,1,8)

            else:
                hyperedge = torch.nonzero(H[i], as_tuple=False)  #i=1, hyperedge (1,1), tensor=2, i=2, tensor=2, hyperedge (1,1)
                hyperedge_num = (hyperedge.squeeze(0)).squeeze(0)  #size 0, tensor 2   indus_ten(32,758,8)    #y4 = all_hyperedge_tensor[hyperedge_num, :, i, :].unsqueeze(1) #32,1,8
                indus_fund = torch.cat([industry_tensor[:, i, :].unsqueeze(1), all_hyperedge_tensor[hyperedge_num, :, i, :].unsqueeze(1)], dim=1)  #32,2,8
                ## coefs = self.attn(indus_fund)    #inter-hypergraph attention?      coefs(32,1,2)
                ## final_indus_fund = torch.matmul(coefs, indus_fund)  #indus_fund (32,2,8),final: (32,1,8)

                y9 =self.conv(indus_fund)  #no need add final_fts_i
                final_tensor = torch.cat([final_tensor, y9], dim=1)   #i=2  final_tensor(32,3,8)

        x = F.dropout(final_tensor, self.dropout, training=self.training)   #32,758,8   32, 1749,8???
        return x



class HGAT(nn.Module):
    """Tri-attention Modules."""
    def __init__(self, nfeat, nhid, dropout):    #nfeat = rnn_unit
        super(HGAT, self).__init__()
        self.hyperedge = hyperedge_attn(nfeat, nhid, dropout) #intra-hyperedge attention
        self.attn = edge_attn(nhid, nhid*2)    #from layers   inter-hyperedge attention
        self.dropout = dropout

    def forward(self, x, H, adj,nhid):    #x(32,758,32) is enc_ooutput
        x = F.dropout(x, self.dropout, training=self.training)   #x(32,758,32)  H(758,62) nhid 8 adj(758,758)
        all_hyperedge_tensor, hyperedge_tensor, industry_tensor = self.hyperedge(x, H, adj, nhid) #all(62,32,758,8), hyp(32,62,8) indus_ten(32,758,8)
        #all_hyperedge_fts, hyperedge_fts, industry_tensor  # all_ and hyper all included the fund information
        # all (62,32,758,8)   hyperedge (32,62,8)--after max pooling, industry(32,758,8)
        all_hyperedge_tensor = F.elu(all_hyperedge_tensor)   #(62,32,758,8)
        hyperedge_tensor = F.elu(hyperedge_tensor)
        final_tensor = torch.randn(0).cuda()   #start from 0

        for i in range(x.shape[1]):#758
            final_fts_i = torch.randn(0).cuda()
            hyperedge_fts_i = torch.randn(0).cuda()
            if torch.sum(H, 1)[i] > 1:
                # vertex degree > 1   #hyperedge(2,1), 2 is two nonzero, 1 is input dim ([[0],[1]]) are the index of nonzero of i=0
                hyperedge = torch.nonzero(H[i], as_tuple=False)  #check this shape, view, nonzero return
                for j in range(len(hyperedge)): #i=0, len=2
                    hyperedge_num = hyperedge[j] #[0]
                    #y0 = all_hyperedge_tensor[hyperedge_num, :, i, :] #i,j=0 (1,32,8), no consider of 758 dim
                    #y1 = all_hyperedge_tensor[hyperedge_num, :, i, :].permute(1, 0, 2) #(32,1,8)
                    final_fts_i = torch.cat([final_fts_i, all_hyperedge_tensor[hyperedge_num, :, i, :].permute(1, 0, 2)], dim=1)#all_hyperedge(62,32,758,8)
                    #final_fts_i (32,1,8), cat on 1 dim which is fund dim, after cat on dim 1, become (32,2,8)
                    #y3 = hyperedge_tensor[:, hyperedge_num, :]  #(32,1,8) also start from fund
                    hyperedge_fts_i = torch.cat([hyperedge_fts_i, hyperedge_tensor[:, hyperedge_num, :]], dim=1) #hyperedge_tensor (32,62,8) check the view

                coefs = self.attn(hyperedge_fts_i) #inter-hyperedge attention, i=0,j=1 coefs (32,1,2)  hyperedge_fts(32,2,8)   final_fts_i(32,2,8)
                final_fts_i = torch.matmul(coefs, final_fts_i) #i=0, final_fts_i (32,2,8)->(32,1,8)  #between hyperedge_fts_i and final_fts_i
                indus_fund = torch.cat([industry_tensor[:, i, :].unsqueeze(1), final_fts_i], dim=1) #indus_ten(32,758,8)   -> indus_fund(32,2,8)

                coefs = self.attn(indus_fund) #inter-hyperedge attention or inter-hypergraph attention? coefs(32,1,2)
                final_indus_fund = torch.matmul(coefs, indus_fund) #final: (32,1,8)
                final_tensor = torch.cat([final_tensor, final_indus_fund], dim=1)  #(32,1,8)

            else:
                hyperedge = torch.nonzero(H[i], as_tuple=False)  #i=1, hyperedge (1,1), tensor=2, i=2, tensor=2, hyperedge (1,1)
                hyperedge_num = (hyperedge.squeeze(0)).squeeze(0)  #size 0, tensor 2   indus_ten(32,758,8)
                #y4 = all_hyperedge_tensor[hyperedge_num, :, i, :].unsqueeze(1) #32,1,8
                indus_fund = torch.cat([industry_tensor[:, i, :].unsqueeze(1), all_hyperedge_tensor[hyperedge_num, :, i, :].unsqueeze(1)], dim=1)  #32,2,8
                coefs = self.attn(indus_fund)    #inter-hypergraph attention?      coefs(32,1,2)
                final_indus_fund = torch.matmul(coefs, indus_fund)  #indus_fund (32,2,8),final: (32,1,8)
                final_tensor = torch.cat([final_tensor, final_indus_fund], dim=1)   #i=2  final_tensor(32,3,8)

        x = F.dropout(final_tensor, self.dropout, training=self.training)   #32,758,8
        return x


"""H[0] 62 [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.]
    
    H[1]   [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.]
    H[2] [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.]
    
    H3 [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.]
        

"""
class HU(nn.Module):
    """Tri-attention Modules."""
    def __init__(self, nfeat, nhid, dropout):    #nfeat = rnn_unit
        super(HU, self).__init__()
        self.hyperedge = hyperedge_attn(nfeat, nhid, dropout) #intra-hyperedge attention
        self.attn = edge_attn(nhid, nhid*2)    #from layers   inter-hyperedge attention
        self.conv = EdgeConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, H, adj,nhid):    #x(32,758,32) is enc_ooutput
        x = F.dropout(x, self.dropout, training=self.training)   #x(32,758,32)  H(758,62) nhid 8 adj(758,758)
        all_hyperedge_tensor, hyperedge_tensor, industry_tensor = self.hyperedge(x, H, adj, nhid) #all(62,32,758,8), hyp(32,62,8) indus_ten(32,758,8)
        #all_hyperedge_fts, hyperedge_fts, industry_tensor  # all_ and hyper all included the fund information
        # all (62,32,758,8)   hyperedge (32,62,8)--after max pooling, industry(32,758,8)
        all_hyperedge_tensor = F.elu(all_hyperedge_tensor)   #(62,32,758,8)
        hyperedge_tensor = F.elu(hyperedge_tensor)
        final_tensor = torch.randn(0).cuda()   #start from 0

        for i in range(x.shape[1]):#758
            final_fts_i = torch.randn(0).cuda()
            hyperedge_fts_i = torch.randn(0).cuda()
            if torch.sum(H, 1)[i] > 1:
                # vertex degree > 1   #hyperedge(2,1), 2 is two nonzero, 1 is input dim ([[0],[1]]) are the index of nonzero of i=0
                hyperedge = torch.nonzero(H[i], as_tuple=False)  #check this shape, view, nonzero return
                for j in range(len(hyperedge)): #i=0, len=2
                    hyperedge_num = hyperedge[j] #[0]
                    #y0 = all_hyperedge_tensor[hyperedge_num, :, i, :] #i,j=0 (1,32,8), no consider of 758 dim
                    #y1 = all_hyperedge_tensor[hyperedge_num, :, i, :].permute(1, 0, 2) #(32,1,8)
                    final_fts_i = torch.cat([final_fts_i, all_hyperedge_tensor[hyperedge_num, :, i, :].permute(1, 0, 2)], dim=1)#all_hyperedge(62,32,758,8)
                    #final_fts_i (32,1,8), cat on 1 dim which is fund dim, after cat on dim 1, become (32,2,8)
                    #y3 = hyperedge_tensor[:, hyperedge_num, :]  #(32,1,8) also start from fund
                    hyperedge_fts_i = torch.cat([hyperedge_fts_i, hyperedge_tensor[:, hyperedge_num, :]], dim=1) #hyperedge_tensor (32,62,8) check the view

                ## coefs = self.attn(hyperedge_fts_i) #inter-hyperedge attention, i=0,j=1 coefs (32,1,2)
                ## final_fts_i = torch.matmul(coefs, final_fts_i) #i=0, final_fts_i (32,2,8)->(32,1,8)

                y8 = self.conv(final_fts_i)  #add final, delete hyperedge_fts_i
                #'indus_fund = torch.cat([industry_tensor[:, i, :].unsqueeze(1), y8], dim=1) #indus_ten(32,758,8)   -> indus_fund(32,2,8)
                ###indus_fund = industry_tensor[:, i, :].unsqueeze(1) + y8
                ## coefs = self.attn(indus_fund) #inter-hyperedge attention or inter-hypergraph attention? coefs(32,1,2)
                ## final_indus_fund = torch.matmul(coefs, indus_fund) #final: (32,1,8)
                  ###
                #'y9 = self.conv(indus_fund)  #add final   #####can check more 1230 important, with two indus_fund  add final, delete hyperedge_fts_i
                final_tensor = torch.cat([final_tensor, y8], dim=1)  #(32,1,8)

            else:
                hyperedge = torch.nonzero(H[i], as_tuple=False)  #i=1, hyperedge (1,1), tensor=2, i=2, tensor=2, hyperedge (1,1)
                hyperedge_num = (hyperedge.squeeze(0)).squeeze(0)  #size 0, tensor 2   indus_ten(32,758,8)    #y4 = all_hyperedge_tensor[hyperedge_num, :, i, :].unsqueeze(1) #32,1,8
                indus_fund = torch.cat([industry_tensor[:, i, :].unsqueeze(1), all_hyperedge_tensor[hyperedge_num, :, i, :].unsqueeze(1)], dim=1)  #32,2,8


                ## coefs = self.attn(indus_fund)    #inter-hypergraph attention?      coefs(32,1,2)
                ## final_indus_fund = torch.matmul(coefs, indus_fund)  #indus_fund (32,2,8),final: (32,1,8)

                y9 =self.conv(indus_fund)  #no need add final_fts_i
                final_tensor = torch.cat([final_tensor, y9], dim=1)   #i=2  final_tensor(32,3,8)

        x = F.dropout(final_tensor, self.dropout, training=self.training)   #32,758,8   32, 1749,8???
        return x


class HGN(nn.Module):
    """Tri-attention Modules."""
    def __init__(self, nfeat, nhid, dropout):    #nfeat = rnn_unit
        super(HGN, self).__init__()
        self.hyperedge = hyperedge_attn(nfeat, nhid, dropout) #intra-hyperedge attention
        self.attn = edge_attn(nhid, nhid*2)    #from layers   inter-hyperedge attention
        self.conv = EdgeConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, H, adj,nhid):    #x(32,758,32) is enc_ooutput
        x = F.dropout(x, self.dropout, training=self.training)   #x(32,758,32)  H(758,62) nhid 8 adj(758,758)
        all_hyperedge_tensor, hyperedge_tensor, industry_tensor = self.hyperedge(x, H, adj, nhid) #all(62,32,758,8), hyp(32,62,8) indus_ten(32,758,8)
        #all_hyperedge_fts, hyperedge_fts, industry_tensor  # all_ and hyper all included the fund information
        # all (62,32,758,8)   hyperedge (32,62,8)--after max pooling, industry(32,758,8)
        all_hyperedge_tensor = F.elu(all_hyperedge_tensor)   #(62,32,758,8)
        hyperedge_tensor = F.elu(hyperedge_tensor)
        final_tensor = torch.randn(0).cuda()   #start from 0

        for i in range(x.shape[1]):#758
            final_fts_i = torch.randn(0).cuda()
            hyperedge_fts_i = torch.randn(0).cuda()
            if torch.sum(H, 1)[i] > 1:
                # vertex degree > 1   #hyperedge(2,1), 2 is two nonzero, 1 is input dim ([[0],[1]]) are the index of nonzero of i=0
                hyperedge = torch.nonzero(H[i], as_tuple=False)  #check this shape, view, nonzero return
                for j in range(len(hyperedge)): #i=0, len=2
                    hyperedge_num = hyperedge[j] #[0]
                    #y0 = all_hyperedge_tensor[hyperedge_num, :, i, :] #i,j=0 (1,32,8), no consider of 758 dim
                    #y1 = all_hyperedge_tensor[hyperedge_num, :, i, :].permute(1, 0, 2) #(32,1,8)
                    final_fts_i = torch.cat([final_fts_i, all_hyperedge_tensor[hyperedge_num, :, i, :].permute(1, 0, 2)], dim=1)#all_hyperedge(62,32,758,8)
                    #final_fts_i (32,1,8), cat on 1 dim which is fund dim, after cat on dim 1, become (32,2,8)
                    #y3 = hyperedge_tensor[:, hyperedge_num, :]  #(32,1,8) also start from fund
                    hyperedge_fts_i = torch.cat([hyperedge_fts_i, hyperedge_tensor[:, hyperedge_num, :]], dim=1) #hyperedge_tensor (32,62,8) check the view

                ## coefs = self.attn(hyperedge_fts_i) #inter-hyperedge attention, i=0,j=1 coefs (32,1,2)
                ## final_fts_i = torch.matmul(coefs, final_fts_i) #i=0, final_fts_i (32,2,8)->(32,1,8)

                y8 = self.conv(final_fts_i)  #add final, delete hyperedge_fts_i
                indus_fund = torch.cat([industry_tensor[:, i, :].unsqueeze(1), y8], dim=1) #indus_ten(32,758,8)   -> indus_fund(32,2,8)
                ###indus_fund = industry_tensor[:, i, :].unsqueeze(1) + y8
                ## coefs = self.attn(indus_fund) #inter-hyperedge attention or inter-hypergraph attention? coefs(32,1,2)
                ## final_indus_fund = torch.matmul(coefs, indus_fund) #final: (32,1,8)
                  ###
                y9 = self.conv(indus_fund)  #add final   #####can check more 1230 important, with two indus_fund  add final, delete hyperedge_fts_i
                final_tensor = torch.cat([final_tensor, y9], dim=1)  #(32,1,8)

            else:
                hyperedge = torch.nonzero(H[i], as_tuple=False)  #i=1, hyperedge (1,1), tensor=2, i=2, tensor=2, hyperedge (1,1)
                hyperedge_num = (hyperedge.squeeze(0)).squeeze(0)  #size 0, tensor 2   indus_ten(32,758,8)    #y4 = all_hyperedge_tensor[hyperedge_num, :, i, :].unsqueeze(1) #32,1,8
                indus_fund = torch.cat([industry_tensor[:, i, :].unsqueeze(1), all_hyperedge_tensor[hyperedge_num, :, i, :].unsqueeze(1)], dim=1)  #32,2,8

                ## coefs = self.attn(indus_fund)    #inter-hypergraph attention?      coefs(32,1,2)
                ## final_indus_fund = torch.matmul(coefs, indus_fund)  #indus_fund (32,2,8),final: (32,1,8)

                y9 =self.conv(indus_fund)  #no need add final_fts_i
                final_tensor = torch.cat([final_tensor, y9], dim=1)   #i=2  final_tensor(32,3,8)

        x = F.dropout(final_tensor, self.dropout, training=self.training)   #32,758,8   32, 1749,8???
        return x










class HGAT1(nn.Module):
    """Tri-attention Modules."""
    def __init__(self, nfeat, nhid, dropout):
        super(HGAT1, self).__init__()
        self.hyperedge = hyperedge_attn(nfeat, nhid, dropout)  # hyperedge
        self.attn = edge_attn(nhid, nhid * 2)  # from layers
        self.dropout = dropout

    def forward(self, x, H, adj, nhid):  # here
        x = F.dropout(x, self.dropout, training=self.training)  # x(32,758,32)  H(758,62) nhid 8 adj(758,758)

        all_hyperedge_tensor, hyperedge_tensor, industry_tensor = self.hyperedge(x, H, adj,
                                                                                 nhid)  # all(62,32,758,8), hyp(32,62,8) indus(32,758,8)

        all_hyperedge_tensor = F.elu(all_hyperedge_tensor)
        hyperedge_tensor = F.elu(hyperedge_tensor)

        final_tensor = torch.randn(0).cuda()  # start from 0

        for i in range(x.shape[1]):
            final_fts_i = torch.randn(0).cuda()
            hyperedge_fts_i = torch.randn(0).cuda()

            if torch.sum(H, 1)[i] > 1:
                # vertex degree > 1
                hyperedge = torch.nonzero(H[i], as_tuple=False)

                for j in range(len(hyperedge)):
                    hyperedge_num = hyperedge[j]
                    final_fts_i = torch.cat(
                        [final_fts_i, all_hyperedge_tensor[hyperedge_num, :, i, :].permute(1, 0, 2)], dim=1)
                    hyperedge_fts_i = torch.cat([hyperedge_fts_i, hyperedge_tensor[:, hyperedge_num, :]], dim=1)

                coefs = self.attn(hyperedge_fts_i)

                #final_fts_i = torch.matmul(coefs, final_fts_i)
                #indus_fund = torch.cat([industry_tensor[:, i, :].unsqueeze(1), final_fts_i], dim=1)
                indus_fund = industry_tensor[:, i, :].unsqueeze(1)

                coefs = self.attn(indus_fund)
                final_indus_fund = torch.matmul(coefs, indus_fund)
                final_tensor = torch.cat([final_tensor, final_indus_fund], dim=1)

            else:
                hyperedge = torch.nonzero(H[i], as_tuple=False)
                hyperedge_num = (hyperedge.squeeze(0)).squeeze(0)
                # indus_fund = torch.cat([industry_tensor[:, i, :].unsqueeze(1),
                #                         all_hyperedge_tensor[hyperedge_num, :, i, :].unsqueeze(1)], dim=1)

                indus_fund = industry_tensor[:, i, :].unsqueeze(1)

                coefs = self.attn(indus_fund)
                final_indus_fund = torch.matmul(coefs, indus_fund)
                final_tensor = torch.cat([final_tensor, final_indus_fund], dim=1)

        x = F.dropout(final_tensor, self.dropout, training=self.training)

        return x


