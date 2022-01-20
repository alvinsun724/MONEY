''' Define the HGTAN model '''
import torch
import torch.nn as nn
from tri_attn import HGAT, HGAT1, HGN, HU, HGN_AD
from temp_layers import TemporalAttention
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Parameter
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import remove_self_loops, add_self_loops
# from torch_geometric.utils import get_laplacian
import math
from hyperedge_attn import BernNet
from torch.nn.modules.module import Module
#from ende import EncoderRNN, Attn, LuongAttnDecoderRNN   #   5
device = torch.device('cuda')

class Mlp(nn.Module):
    def __init__(self, input_dim, hid_dim, n_class):
        super(Mlp, self).__init__()
        self.fc1 = Linear(32, hid_dim)
        self.fc2 = Linear(hid_dim, n_class)
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

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size(0), seq.size(1) #sz_b = 24256, len_s =10
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)  #(10,10)
    subsequent_mask = subsequent_mask.bool().bool().unsqueeze(0).expand(sz_b, -1, -1)
    return subsequent_mask   #(24256,10,10)

def indu_edge_index(adjacency):
    edge_index = torch.nonzero(adjacency, as_tuple=False).permute(1,0) #9826,2
    return edge_index

class DGCN_HGN_AD(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(self, rnn_unit, n_hid, n_class, feature, d_word_vec, d_model,n_head, d_k, d_v, dropout, tgt_emb_prj_weight_sharing, hinge=0):
# d_model:16  d_word_vec: 16 d_k:8  d_v:8 n_head:4 feature:10   n_class:3  n_hid:8  rnn:32 sharing:true, add hinge
        super().__init__()
        self.linear = nn.Linear(feature, d_word_vec)
        self.rnn = nn.GRU(d_word_vec,rnn_unit,num_layers=2,batch_first=True,    #hidden_units in GRU default 32
                          bidirectional=False)       #input_size: number of features of input, hidden_size: number of features in hidden state h, number of recurrent layers,
        self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)    #for get gru
        self.gcn = GCN(feature, n_hid, n_class, dropout)   #self.HGAT = HGAT(nfeat=rnn_unit,nhid=n_hid,dropout=dropout) self.HGN = HGN(nfeat=rnn_unit,nhid=n_hid,dropout=dropout)
        self.HGN_AD = HGN_AD(nfeat=rnn_unit,nhid=n_hid,dropout=dropout, hinge=1) #self.gcn_f = GCN_f(feature, dropout, fd=62)
        self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \ the dimensions of all module outputs shall be the same.'
        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.
        if hinge == 1:
            self.hinge = True
        if hinge == 0:
            self.hinge = False

        self.av_w = Parameter(torch.FloatTensor(n_hid, rnn_unit)) #change in_ to n_hid from hidden as tensordot dim 1 should be same
        self.av_b = Parameter(torch.FloatTensor(rnn_unit))
        self.av_u = Parameter(torch.FloatTensor(rnn_unit))
        self.fc_W = Parameter(torch.FloatTensor(2*n_hid, 3)) #as self.fea_con(24256, 16), *(16,3)-> 24256, 3
        self.fc_b = Parameter(torch.FloatTensor(3))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.av_w.size(1))
        self.av_w.data.uniform_(-stdv, stdv)
        self.av_b.data.uniform_(-stdv, stdv)
        self.av_u.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)


    def adv_part(self, adv_inputs, hidden):
        print('adversial part')   #adv_input(24256, 16)
        # self.fc_W = Parameter(torch.FloatTensor(hidden*2, 1))
        # self.fc_b = Parameter(torch.FloatTensor(1)) #again if set the parameter here error:  Expected all tensors to be on the same device, but found at least two devices,
        if self.hinge:
            out = torch.matmul(adv_inputs, self.fc_W) #use matmul as x is 3 dim, weight is 2 dim
            out = out + self.fc_b
        else:
            out = torch.matmul(self.fea_con, self.fc_W)  # use matmul as x is 3 dim, weight is 2 dim  self.fea_con (24256, 16)
            out = out + self.fc_b
            # m = nn.Sigmoid
            # out = m(out)
        return out  #(24256, 3)


    def construct_graph(self, x, n_hid, hidden):
        # self.av_w = Parameter(torch.FloatTensor(n_hid, hidden)) #change in_ to n_hid from hidden as tensordot dim 1 should be same
        # self.av_b = Parameter(torch.FloatTensor(hidden))
        # self.av_u = Parameter(torch.FloatTensor(hidden))    all these in init then no problem for different device, cpu and cuda!!indeed!!
        # stdv = 1. / math.sqrt(self.av_w.size(1))
        # self.av_w.data.uniform_(-stdv, stdv)
        # self.av_b.data.uniform_(-stdv, stdv)
        # self.av_u.data.uniform_(-stdv, stdv)
        L = Linear(1,x.shape[1]).to(device)

        self.a_laten = F.tanh(torch.tensordot(x, self.av_w, 1) + self.av_b)  #change self.output to x(24256, 8) av_w (32,32)
        self.a_scores = torch.tensordot(self.a_laten, self.av_u, 1)
        self.a_alphas = F.softmax(self.a_scores)
        self.a_con = torch.sum(x * self.a_alphas.unsqueeze(-1), 1)  #change self.output to x
        #self.fea_con = torch.cat([x[:,-1,:], self.a_con],1) #maybe not :,-1,: depend on the output, change self.output to x
        # new_fea = []
        # for i in range(x.shape[1]):
        #     x0 = x[:,i]
        #     x1 = torch.cat(x0, self.a_con)
        # new_fea.append(x1)
        # c = torch.stack(new_fea, dim=1)
        self.a_con = L(self.a_con.unsqueeze(1))
        self.fea_con = torch.cat([x, self.a_con], 1)
        return self.fea_con


    def forward(self, src_seq, H, adj, n_hid):  # src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
        new_src = []
        m = Linear(H.shape[1],H.shape[0]).to(device)
        H_new = m(H)
        for i in range(src_seq.shape[0]):
            x0, x_, x_f = src_seq[i, :, :, :], [], []  # x0 (758,10,10)
            for i in range(x0.shape[1]):
                x1 = x0[:, i, :]  # (758, 10)
                x2 = self.gcn(x1, adj) # x2(758, 10)
                x_.append(x2)
                y_i = torch.stack(x_, dim=1)  # y(758, 10, 10)
                x_fund = self.gcn(x1,H_new) ###
                x_f.append(x_fund)
                y_f = torch.stack(x_f, dim=1)
                y = torch.add(y_i, y_f)
            new_src.append(y)
            z = torch.stack(new_src, dim=0)  # stack input tensors  (32, 758, 10,10)
        # new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
        src_seq = self.linear(z)  # z(32, 758, 10,10)  # src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
        batch = src_seq.size(0)
        stock = src_seq.size(1)
        seq_len = src_seq.size(2)
        dim = src_seq.size(3)
        src_seq = torch.reshape(src_seq, (batch * stock, seq_len, dim))  # (24256, 10, 16)

        slf_attn_mask = get_subsequent_mask(src_seq).bool()
        rnn_output, *_ = self.rnn(src_seq)  # output, hidden state      rnn_out tensor (24256, 10, 32)
        enc_output, enc_slf_attn = self.temp_attn(rnn_output, rnn_output, rnn_output,
            mask=slf_attn_mask.bool())  # enc_output (24256, 32), enc_slf(97024, 10, 10)

        enc_output = torch.reshape(enc_output, (batch, stock, -1))  # (32, 758, 32)  # HGAT_output = self.HGAT(enc_output, H, adj, n_hid)  # here in HGAT_out (32,758,8) , stock embedding update to 8
        HGN_output  = self.HGN_AD(enc_output, H, adj, n_hid)  #HGN(32,1749,8)
        HGN_output = torch.reshape(HGN_output, (batch * stock, -1))  # (24256, 8)

        z = self.construct_graph(HGN_output, n_hid, hidden=4*n_hid)  #extra

        self.pred = self.adv_part(z, hidden=4*n_hid)  #extra   need check the shape

        seq_logit = self.tgt_word_prj(HGN_output) * self.x_logit_scale

        return seq_logit, self.pred  # (24256, 3)   #extra sel.pred


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat) #change as want the same dim
        self.dropout = dropout

    def forward(self, x, adj): #x(758,10) adj(758,758)
        x = F.relu(self.gc1(x, adj)) #x(758,8)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)    #(758,10)
        return x


class DGCN_HGN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(self, rnn_unit, n_hid, n_class, feature, d_word_vec, d_model,n_head, d_k, d_v, dropout, tgt_emb_prj_weight_sharing):
# d_model:16  d_word_vec: 16 d_k:8  d_v:8 n_head:4 feature:10   n_class:3  n_hid:8  rnn:32 sharing:true
        super().__init__()
        self.linear = nn.Linear(feature, d_word_vec)
        self.rnn = nn.GRU(d_word_vec,rnn_unit,num_layers=2,batch_first=True,
                          bidirectional=False)       #input_size: number of features of input, hidden_size: number of features in hidden state h, number of recurrent layers,
        self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)    #for get gru
        self.gcn = GCN(feature, n_hid, n_class, dropout)   #self.HGAT = HGAT(nfeat=rnn_unit,nhid=n_hid,dropout=dropout)
        self.HGN = HGN(nfeat=rnn_unit,nhid=n_hid,dropout=dropout)
        self.gcn_f = GCN_f(feature, dropout, fd=62)
        self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'
        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_seq, H, adj, n_hid):  # src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
        new_src = []
        m = Linear(H.shape[1],H.shape[0]).to(device)
        H_new = m(H)
        for i in range(src_seq.shape[0]):
            x0, x_, x_f = src_seq[i, :, :, :], [], []  # x0 (758,10,10)
            for i in range(x0.shape[1]):
                x1 = x0[:, i, :]  # (758, 10)
                x2 = self.gcn(x1, adj) # x2(758, 10)
                x_.append(x2)
                y_i = torch.stack(x_, dim=1)  # y(758, 10, 10)
                x_fund = self.gcn(x1,H_new) ###
                x_f.append(x_fund)
                y_f = torch.stack(x_f, dim=1)
                y = torch.add(y_i, y_f)
            new_src.append(y)
            z = torch.stack(new_src, dim=0)  # stack input tensors  (32, 758, 10,10)
        # new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
        # new_src=torch.FloatTensor(new_src)
        src_seq = self.linear(z)  # z(32, 758, 10,10)
        # src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
        batch = src_seq.size(0)
        stock = src_seq.size(1)
        seq_len = src_seq.size(2)
        dim = src_seq.size(3)
        src_seq = torch.reshape(src_seq, (batch * stock, seq_len, dim))  # (24256, 10, 16)

        slf_attn_mask = get_subsequent_mask(src_seq).bool()
        # gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
        rnn_output, *_ = self.rnn(src_seq)  # output, hidden state      rnn_out tensor (24256, 10, 32)
        enc_output, enc_slf_attn = self.temp_attn(rnn_output, rnn_output, rnn_output,
            mask=slf_attn_mask.bool())  # enc_output (24256, 32), enc_slf(97024, 10, 10)

        enc_output = torch.reshape(enc_output, (batch, stock, -1))  # (32, 758, 32)
        # HGAT_output = self.HGAT(enc_output, H, adj, n_hid)  # here in HGAT_out (32,758,8) , stock embedding update to 8
        HGN_output = self.HGN(enc_output, H, adj, n_hid)  #HGN(32,1749,8)
        HGN_output = torch.reshape(HGN_output, (batch * stock, -1))  # (24256, 8)
        seq_logit = self.tgt_word_prj(HGN_output) * self.x_logit_scale

        return seq_logit  # (24256, 3)




class DGCN_HGTAN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(self, rnn_unit, n_hid, n_class, feature, d_word_vec, d_model,n_head, d_k, d_v, dropout, tgt_emb_prj_weight_sharing):
# d_model:16  d_word_vec: 16 d_k:8  d_v:8 n_head:4 feature:10   n_class:3  n_hid:8  rnn:32 sharing:true
        super().__init__()
        self.linear = nn.Linear(feature, d_word_vec)
        self.rnn = nn.GRU(d_word_vec,rnn_unit,num_layers=2,batch_first=True,
                          bidirectional=False)       #input_size: number of features of input, hidden_size: number of features in hidden state h, number of recurrent layers,
        self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)    #for get gru
        self.gcn = GCN(feature, n_hid, n_class, dropout)   #self.HGAT = HGAT(nfeat=rnn_unit,nhid=n_hid,dropout=dropout)
        self.HGN = HGN(nfeat=rnn_unit,nhid=n_hid,dropout=dropout)
        self.gcn_f = GCN_f(feature, dropout, fd=62)
        self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'
        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_seq, H, adj, n_hid):  # src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
        new_src = []
        m = Linear(H.shape[1],H.shape[0]).to(device)
        H_new = m(H)
        for i in range(src_seq.shape[0]):
            x0, x_, x_f = src_seq[i, :, :, :], [], []  # x0 (758,10,10)
            for i in range(x0.shape[1]):
                x1 = x0[:, i, :]  # (758, 10)
                x2 = self.gcn(x1, adj) # x2(758, 10)
                x_.append(x2)
                y_i = torch.stack(x_, dim=1)  # y(758, 10, 10)
                x_fund = self.gcn(x1,H_new) ###
                x_f.append(x_fund)
                y_f = torch.stack(x_f, dim=1)

                y = torch.add(y_i, y_f)
            new_src.append(y)
            z = torch.stack(new_src, dim=0)  # stack input tensors  (32, 758, 10,10)
            # new_src.append(x_)
        # new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
        # new_src=torch.FloatTensor(new_src)
        src_seq = self.linear(z)  # z(32, 758, 10,10)
        # src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
        batch = src_seq.size(0)
        stock = src_seq.size(1)
        seq_len = src_seq.size(2)
        dim = src_seq.size(3)
        src_seq = torch.reshape(src_seq, (batch * stock, seq_len, dim))  # (24256, 10, 16)
        slf_attn_mask = get_subsequent_mask(src_seq).bool()
        # gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
        rnn_output, *_ = self.rnn(src_seq)  # output, hidden state      rnn_out tensor (24256, 10, 32)
        enc_output, enc_slf_attn = self.temp_attn(rnn_output, rnn_output, rnn_output, mask=slf_attn_mask.bool())  # enc_output (24256, 32), enc_slf(97024, 10, 10)

        enc_output = torch.reshape(enc_output, (batch, stock, -1))  # (32, 758, 32)

        # HGAT_output = self.HGAT(enc_output, H, adj, n_hid)  # here in HGAT_out (32,758,8) , stock embedding update to 8
        HGAT_output = self.HGN(enc_output, H, adj, n_hid)  #HGN(32,1749,8)
        HGAT_output = torch.reshape(HGAT_output, (batch * stock, -1))  # (24256, 8)
        seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale

        return seq_logit  # (24256, 3)




class DGCN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(self, rnn_unit, n_hid, n_class, feature, d_word_vec, d_model,n_head, d_k, d_v, dropout, tgt_emb_prj_weight_sharing):
# d_model:16  d_word_vec: 16 d_k:8  d_v:8 n_head:4 feature:10   n_class:3  n_hid:8  rnn:32 sharing:true
        super().__init__()
        self.linear = nn.Linear(feature, d_word_vec)
        self.rnn = nn.GRU(d_word_vec,rnn_unit,num_layers=2,batch_first=True,
                          bidirectional=False)       #input_size: number of features of input, hidden_size: number of features in hidden state h, number of recurrent layers,
        self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)    #for get gru
        self.gcn = GCN(feature, n_hid, n_class, dropout)   #self.HGAT = HGAT(nfeat=rnn_unit,nhid=n_hid,dropout=dropout)
        self.HGN = HGN(nfeat=rnn_unit,nhid=n_hid,dropout=dropout)
        self.gcn_f = GCN_f(feature, dropout, fd=62)
        self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'
        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_seq, H, adj, n_hid):  # src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
        new_src = []
        m = Linear(H.shape[1],H.shape[0]).to(device)
        l = Linear(4*n_hid,n_hid).to(device)
        H_new = m(H)
        for i in range(src_seq.shape[0]):
            x0, x_, x_f = src_seq[i, :, :, :], [], []  # x0 (758,10,10)
            for i in range(x0.shape[1]):
                x1 = x0[:, i, :]  # (758, 10)
                x2 = self.gcn(x1, adj) # x2(758, 10)
                x_.append(x2)
                y_i = torch.stack(x_, dim=1)  # y(758, 10, 10)
                x_fund = self.gcn(x1,H_new) ###
                x_f.append(x_fund)
                y_f = torch.stack(x_f, dim=1)
                y = torch.add(y_i, y_f)
            new_src.append(y)
            z = torch.stack(new_src, dim=0)  # stack input tensors  (32, 758, 10,10)
        # new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
        # new_src=torch.FloatTensor(new_src)
        src_seq = self.linear(z)  # z(32, 758, 10,10)
        # src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
        batch = src_seq.size(0)
        stock = src_seq.size(1)
        seq_len = src_seq.size(2)
        dim = src_seq.size(3)
        src_seq = torch.reshape(src_seq, (batch * stock, seq_len, dim))  # (24256, 10, 16)

        slf_attn_mask = get_subsequent_mask(src_seq).bool()
        # gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
        rnn_output, *_ = self.rnn(src_seq)  # output, hidden state      rnn_out tensor (24256, 10, 32)
        enc_output, enc_slf_attn = self.temp_attn(rnn_output, rnn_output, rnn_output,
            mask=slf_attn_mask.bool())  # enc_output (24256, 32), enc_slf(97024, 10, 10)

        enc_output = torch.reshape(enc_output, (batch, stock, -1))  # (32, 758, 32)
        # HGAT_output = self.HGAT(enc_output, H, adj, n_hid)  # here in HGAT_out (32,758,8) , stock embedding update to 8
        #HGN_output = self.HGN(enc_output, H, adj, n_hid)  #HGN(32,1749,8)
        HGN_output = torch.reshape(enc_output , (batch * stock, -1))  # (24256, 8)  (24256,32)
        HGN_output = l(HGN_output)
        seq_logit = self.tgt_word_prj(HGN_output) * self.x_logit_scale

        return seq_logit  # (24256, 3)

class HGTAN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(self, rnn_unit, n_hid, n_class, feature, d_word_vec, d_model, n_head, d_k, d_v, dropout, tgt_emb_prj_weight_sharing):
# d_model:16  d_word_vec: 16 d_k:8  d_v:8 n_head:4 feature:10   n_class:3  n_hid:8  rnn:32 sharing:true
        super().__init__()
        self.linear = nn.Linear(feature, d_word_vec)
        self.rnn = nn.GRU(d_word_vec,rnn_unit, num_layers=2, batch_first=True,
                          bidirectional=False)       #input_size: number of features of input, hidden_size: number of features in hidden state h, number of recurrent layers,
        # batch fist, input and out tensor as (batch, seq, feature)
        self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)   #for get gru

        self.HGAT = HGAT(nfeat=rnn_unit, nhid=n_hid, dropout=dropout)
        self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \  the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_seq, H, adj, n_hid): #src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
        src_seq = self.linear(src_seq)   #(32, 758, 10, 16)
        batch = src_seq.size(0) #32
        stock = src_seq.size(1) #758
        seq_len = src_seq.size(2) #10
        dim = src_seq.size(3) #16     price embedded into 16 dim
        src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim))   #(24256, 10, 16)

        slf_attn_mask = get_subsequent_mask(src_seq).bool()

        rnn_output, *_ = self.rnn(src_seq)   #output, hidden state
        #rnn_out tensor (24256, 10, 32), src_seq(24256,10, 16)
        enc_output, enc_slf_attn = self.temp_attn(
            rnn_output, rnn_output, rnn_output, mask=slf_attn_mask.bool())   #enc_output (24256, 32), enc_slf(97024, 10, 10)

        enc_output = torch.reshape(enc_output, (batch, stock, -1))  #(32, 758, 32)

        HGAT_output = self.HGAT(enc_output, H, adj, n_hid)   #here in HGAT_out (32,758,8) , stock embedding update to 8
        HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))   #(24256, 8)
        seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale

        return seq_logit  #(24256, 3)


class Bern_HG_Fu_Att(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(self, rnn_unit, n_hid, n_class, feature, d_word_vec, d_model, n_head, d_k, d_v, dropout,
                 tgt_emb_prj_weight_sharing):
        # d_model:16  d_word_vec: 16 d_k:8  d_v:8 n_head:4 feature:10   n_class:3  n_hid:8  rnn:32 sharing:true
        super().__init__()
        self.linear = nn.Linear(feature, d_word_vec)
        self.rnn = nn.GRU(d_word_vec,
                          rnn_unit,
                          num_layers=2,
                          batch_first=True,
                          bidirectional=False)  # input_size: number of features of input, hidden_size: number of features in hidden state h, number of recurrent layers,
        self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)  # for get gru
        self.gcn = GCN(feature, n_hid, n_class, dropout)
        self.bern = BernNet(feature, n_hid, n_class,3, 0.5, dropout)
        self.HGAT = HGAT(nfeat=rnn_unit, nhid=n_hid, dropout=dropout)
        self.HGN = HGN(nfeat=rnn_unit, nhid=n_hid, dropout=dropout)
        self.HU = HU(nfeat=rnn_unit, nhid=n_hid, dropout=dropout)

        self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_seq, H, adj, n_hid):  # src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
        edge_index = indu_edge_index(adj)
        new_src = []
        for i in range(src_seq.shape[0]):
            x0, x_ = src_seq[i, :, :, :], []  # x0 (758,10,10)
            for i in range(x0.shape[1]):
                x1 = x0[:, i, :]  # (758, 10)
                #x2 = self.gcn(x1, adj)  # x2(758, 10)
                x2 = self.bern(x1, edge_index)
                x_.append(x2)
                y = torch.stack(x_, dim=1)  # y(758, 10, 10)
            new_src.append(y)
            z = torch.stack(new_src, dim=0)  # stack input tensors  (32, 758, 10,10)
            # new_src.append(x_)
        # new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
        # new_src=torch.FloatTensor(new_src)
        src_seq = self.linear(z)  # z(32, 758, 10,10)
        # src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
        batch = src_seq.size(0)
        stock = src_seq.size(1)
        seq_len = src_seq.size(2)
        dim = src_seq.size(3)
        src_seq = torch.reshape(src_seq, (batch * stock, seq_len, dim))  # (24256, 10, 16)

        slf_attn_mask = get_subsequent_mask(src_seq).bool()
        # gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
        rnn_output, *_ = self.rnn(src_seq)  # output, hidden state      rnn_out tensor (24256, 10, 32)
        enc_output, enc_slf_attn = self.temp_attn(
            rnn_output, rnn_output, rnn_output,
            mask=slf_attn_mask.bool())  # enc_output (24256, 32), enc_slf(97024, 10, 10)

        enc_output = torch.reshape(enc_output, (batch, stock, -1))  # (32, 758, 32)

        # HGAT_output = self.HGAT(enc_output, H, adj, n_hid)  # here in HGAT_out (32,758,8) , stock embedding update to 8
        HGAT_output = self.HU(enc_output, H, adj, n_hid)  # HGN(32,1749,8)
        HGAT_output = torch.reshape(HGAT_output, (batch * stock, -1))  # (24256, 8)
        seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale

        return seq_logit  # (24256, 3)

class Bern_HGN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(self,rnn_unit, n_hid, n_class,feature,d_word_vec, d_model,n_head, d_k, d_v, dropout,tgt_emb_prj_weight_sharing):
# d_model:16  d_word_vec: 16 d_k:8  d_v:8 n_head:4 feature:10   n_class:3  n_hid:8  rnn:32 sharing:true
        super().__init__()
        self.linear = nn.Linear(feature, d_word_vec)
        self.rnn = nn.GRU(d_word_vec,rnn_unit,num_layers=2,batch_first=True,bidirectional=False)       #input_size: number of features of input, hidden_size: number of features in hidden state h, number of recurrent layers,
        self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)    #for get gru
        self.gcn = GCN(feature, n_hid, n_class, dropout)     #self.HGAT = HGAT(nfeat=rnn_unit,nhid=n_hid,dropout=dropout)
        self.bern =BernNet(feature, n_hid, n_class, 10, 0.5, dropout)
        self.HGN = HGN(nfeat=rnn_unit,nhid=n_hid,dropout=dropout)

        self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \ the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_seq, H, adj, n_hid):  # src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
        in_edge_index = indu_edge_index(adj)
        new_src = []
        for i in range(src_seq.shape[0]):
            x0, x_ = src_seq[i, :, :, :], []  # x0 (758,10,10)
            for i in range(x0.shape[1]):
                x1 = x0[:, i, :]  # (758, 10)
                #x2 = self.gcn(x1, adj)  # x2(758, 10)
                x2 = self.bern(x1, in_edge_index)
                x_.append(x2)
                y = torch.stack(x_, dim=1)  # y(758, 10, 10)
            new_src.append(y)
            z = torch.stack(new_src, dim=0)  # stack input tensors  (32, 758, 10,10)
            # new_src.append(x_)
        # new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
        # new_src=torch.FloatTensor(new_src)
        src_seq = self.linear(z)  # z(32, 758, 10,10)
        # src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
        batch = src_seq.size(0)
        stock = src_seq.size(1)
        seq_len = src_seq.size(2)
        dim = src_seq.size(3)
        src_seq = torch.reshape(src_seq, (batch * stock, seq_len, dim))  # (24256, 10, 16)

        slf_attn_mask = get_subsequent_mask(src_seq).bool()
        # gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
        rnn_output, *_ = self.rnn(src_seq)  # output, hidden state      rnn_out tensor (24256, 10, 32)
        enc_output, enc_slf_attn = self.temp_attn( rnn_output, rnn_output, rnn_output, mask=slf_attn_mask.bool())  # enc_output (24256, 32), enc_slf(97024, 10, 10)

        enc_output = torch.reshape(enc_output, (batch, stock, -1))  # (32, 758, 32)

        HGAT_output = self.HGN(enc_output, H, adj, n_hid)      # HGAT_output = self.HGAT(enc_output, H, adj, n_hid)  # here in HGAT_out (32,758,8) , stock embedding update to 8
        HGAT_output = torch.reshape(HGAT_output, (batch * stock, -1))  # (24256, 8)
        seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale
        return seq_logit  # (24256, 3)





class GraphConvolution_f(Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False): #change to no bias
        super(GraphConvolution_f, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support.T)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_f(nn.Module):
    def __init__(self, nfeat, dropout, fd):
        super(GCN_f, self).__init__()
        self.gc1 = GraphConvolution_f(nfeat, fd)
        self.gc2 = GraphConvolution_f(fd, nfeat) #change as want the same dim
        self.dropout = dropout

    def forward(self, x, adj): #x(758,10) H(758,62)
        x = F.relu(self.gc1(x, adj)) #x(758,10)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)    #(758,10)
        return x


class GCN_HGTAN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(self,rnn_unit, n_hid, n_class, feature, d_word_vec, d_model, n_head, d_k, d_v, dropout,tgt_emb_prj_weight_sharing):
        super().__init__()
        self.linear = nn.Linear(feature, d_word_vec)
        self.rnn = nn.GRU(d_word_vec, rnn_unit,num_layers=2, batch_first=True,bidirectional=False)
        self.gcn = GCN(feature, n_hid, n_class, dropout)
        self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)   #for get gru
        self.HGAT = HGAT(nfeat=rnn_unit,nhid=n_hid,dropout=dropout)

        self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \ the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_seq, H, adj, n_hid):   #src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
        new_src = []
        for i in range(src_seq.shape[0]):
            x0, x_ = src_seq[i,:,:,:], []   #x0 (758,10,10)
            for i in range(x0.shape[1]):
                x1 = x0[:,i,:]   #(758, 10)
                x2 = self.gcn(x1, adj) #x2(758, 10)
                x_.append(x2)
                y = torch.stack(x_, dim=1) #y(758, 10, 10)
            new_src.append(y)
            z = torch.stack(new_src, dim=0) #stack input tensors  (32, 758, 10,10)
            #new_src.append(x_)
        #new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
        #new_src=torch.FloatTensor(new_src)
        src_seq = self.linear(z)   #z(32, 758, 10,10)
        #src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
        batch = src_seq.size(0)
        stock = src_seq.size(1)
        seq_len = src_seq.size(2)
        dim = src_seq.size(3)
        src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim)) #(24256, 10, 16)

        slf_attn_mask = get_subsequent_mask(src_seq).bool()
        #gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
        rnn_output, *_ = self.rnn(src_seq)   #output, hidden state      rnn_out tensor (24256, 10, 32)
        enc_output, enc_slf_attn = self.temp_attn(
            rnn_output, rnn_output, rnn_output, mask=slf_attn_mask.bool())   #enc_output (24256, 32), enc_slf(97024, 10, 10)

        enc_output = torch.reshape(enc_output, (batch, stock, -1))  #(32, 758, 32)

        HGAT_output = self.HGAT(enc_output, H, adj, n_hid)   #here in HGAT_out (32,758,8) , stock embedding update to 8
        HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))   #(24256, 8)
        seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale

        return seq_logit  #(24256, 3)


class GCN_HGN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(self, rnn_unit, n_hid, n_class, feature, d_word_vec, d_model,n_head, d_k, d_v, dropout, tgt_emb_prj_weight_sharing):
# d_model:16  d_word_vec: 16 d_k:8  d_v:8 n_head:4 feature:10   n_class:3  n_hid:8  rnn:32 sharing:true
        super().__init__()
        self.linear = nn.Linear(feature, d_word_vec)
        self.rnn = nn.GRU(d_word_vec,rnn_unit,num_layers=2, batch_first=True,
                          bidirectional=False)       #input_size: number of features of input, hidden_size: number of features in hidden state h, number of recurrent layers,
        self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)    #for get gru
        self.gcn = GCN(feature, n_hid, n_class, dropout)
        self.HGAT = HGAT(nfeat=rnn_unit,nhid=n_hid,dropout=dropout)
        self.HGN = HGN(nfeat=rnn_unit,nhid=n_hid,dropout=dropout)

        self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_seq, H, adj, n_hid):  # src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
        new_src = []
        for i in range(src_seq.shape[0]):
            x0, x_ = src_seq[i, :, :, :], []  # x0 (758,10,10)
            for i in range(x0.shape[1]):
                x1 = x0[:, i, :]  # (758, 10)
                x2 = self.gcn(x1, adj)  # x2(758, 10)
                x_.append(x2)
                y = torch.stack(x_, dim=1)  # y(758, 10, 10)
            new_src.append(y)
            z = torch.stack(new_src, dim=0)  # stack input tensors  (32, 758, 10,10)
            # new_src.append(x_)
        # new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
        # new_src=torch.FloatTensor(new_src)
        src_seq = self.linear(z)  # z(32, 758, 10,10)
        # src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
        batch = src_seq.size(0)
        stock = src_seq.size(1)
        seq_len = src_seq.size(2)
        dim = src_seq.size(3)
        src_seq = torch.reshape(src_seq, (batch * stock, seq_len, dim))  # (24256, 10, 16)

        slf_attn_mask = get_subsequent_mask(src_seq).bool()
        # gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
        rnn_output, *_ = self.rnn(src_seq)  # output, hidden state      rnn_out tensor (24256, 10, 32)
        enc_output, enc_slf_attn = self.temp_attn(rnn_output, rnn_output, rnn_output,
            mask=slf_attn_mask.bool())  # enc_output (24256, 32), enc_slf(97024, 10, 10)

        enc_output = torch.reshape(enc_output, (batch, stock, -1))  # (32, 758, 32)

        # HGAT_output = self.HGAT(enc_output, H, adj, n_hid)  # here in HGAT_out (32,758,8) , stock embedding update to 8
        HGAT_output = self.HGN(enc_output, H, adj, n_hid)  #HGN(32,1749,8)
        HGAT_output = torch.reshape(HGAT_output, (batch * stock, -1))  # (24256, 8)
        seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale

        return seq_logit  # (24256, 3)

"""
tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0', dtype=torch.uint8)

subsequent_mask tensor([[[False,  True,  True,  ...,  True,  True,  True],
         [False, False,  True,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         ...,
         [False, False, False,  ..., False,  True,  True],
         [False, False, False,  ..., False, False,  True],
         [False, False, False,  ..., False, False, False]],

        [[False,  True,  True,  ...,  True,  True,  True],
         [False, False,  True,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         ...,
         [False, False, False,  ..., False,  True,  True],
         [False, False, False,  ..., False, False,  True],
         [False, False, False,  ..., False, False, False]],

        [[False,  True,  True,  ...,  True,  True,  True],
         [False, False,  True,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         ...,
         [False, False, False,  ..., False,  True,  True],
         [False, False, False,  ..., False, False,  True],
         [False, False, False,  ..., False, False, False]],
"""





#
#
# class BernNet(torch.nn.Module):
#     def __init__(self,dataset, args):
#         super(BernNet, self).__init__()
#         self.lin1 = Linear(dataset.num_features, args.hidden)
#         self.lin2 = Linear(args.hidden, dataset.num_classes)
#         self.m = torch.nn.BatchNorm1d(dataset.num_classes)
#         self.prop1 = Bern_prop(args.K)
#
#         self.dprate = args.dprate
#         self.dropout = args.dropout
#
#     def reset_parameters(self):
#         self.prop1.reset_parameters()
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.lin2(x)
#         #x= self.m(x)
#         if self.dprate == 0.0:
#             x = self.prop1(x, edge_index)
#             return F.log_softmax(x, dim=1)
#         else:
#             x = F.dropout(x, p=self.dprate, training=self.training)
#             x = self.prop1(x, edge_index)
#             return F.log_softmax(x, dim=1)
#
# class Bern_prop(Module):
#     def __init__(self, K, bias=True, **kwargs):  #propgation steps
#         super(Bern_prop, self).__init__(aggr='add', **kwargs)
#         self.K = K
#         self.temp = Parameter(torch.Tensor(self.K + 1))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.temp.data.fill_(1)
#
#     def forward(self, x, edge_index, edge_weight=None):
#         TEMP = F.relu(self.temp)
#         # L=I-D^(-0.5)AD^(-0.5)
#         edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
#                                            num_nodes=x.size(self.node_dim))
#         # 2I-L
#         edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))
#
#         tmp = []
#         tmp.append(x)
#         for i in range(self.K):
#             x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
#             tmp.append(x)
#
#         out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]
#
#         for i in range(self.K):
#             x = tmp[self.K - i - 1]
#             x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
#             for j in range(i):
#                 x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
#
#             out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x
#         return out
#
#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j
#
#     def __repr__(self):
#         return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
#                                           self.temp)




# class GCN_attn(nn.Module):
#     ''' A sequence to sequence model with attention mechanism. '''
#     def __init__(
#             self,
#             rnn_unit, n_hid, n_class,
#             feature,
#             d_word_vec, d_model,
#             n_head, d_k, d_v, dropout,
#             tgt_emb_prj_weight_sharing):
#         super().__init__()
#         self.linear = nn.Linear(feature, d_word_vec)
#         self.rnn = nn.GRU(d_word_vec,
#                           rnn_unit,
#                           num_layers=2,
#                           batch_first=True,
#                           bidirectional=False)
#         self.gcn = GCN(feature, n_hid, n_class, dropout)
#         self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)   #for get gru
#         self.HGAT = HGAT(nfeat=rnn_unit,
#                          nhid=n_hid,
#                          dropout=dropout,
#                          )
#
#         self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
#         self.classifier = Mlp(feature, n_hid, n_class)
#         nn.init.xavier_normal_(self.tgt_word_prj.weight)
#
#         assert d_model == d_word_vec, \
#         'To facilitate the residual connections, \
#          the dimensions of all module outputs shall be the same.'
#
#         if tgt_emb_prj_weight_sharing:
#             # Share the weight matrix between target word embedding & the final logit dense layer
#             self.x_logit_scale = (d_model ** -0.5)
#         else:
#             self.x_logit_scale = 1.
#
#     def forward(self, src_seq, H, adj, n_hid):   #src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
#         new_src = []
#         for i in range(src_seq.shape[0]):
#             x0, x_ = src_seq[i,:,:,:], []   #x0 (758,10,10)
#             for i in range(x0.shape[1]):
#                 x1 = x0[:,i,:]   #(758, 10)
#                 x2 = self.gcn(x1, adj) #x2(758, 10)
#                 x_.append(x2)
#                 y = torch.stack(x_, dim=1) #y(758, 10, 10)
#             new_src.append(y)
#             z = torch.stack(new_src, dim=0) #stack input tensors  (32, 758, 10,10)
#
#         src_seq = self.linear(z)   #z(32, 758, 10,10)
#         batch = src_seq.size(0)
#         stock = src_seq.size(1)
#         seq_len = src_seq.size(2)
#         dim = src_seq.size(3)
#         src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim)) #(24256, 10, 16)
#
#         slf_attn_mask = get_subsequent_mask(src_seq).bool()
#         rnn_output, *_ = self.rnn(src_seq)   #output, hidden state      rnn_out tensor (24256, 10, 32)
#         enc_output, enc_slf_attn = self.temp_attn(
#             rnn_output, rnn_output, rnn_output, mask=slf_attn_mask.bool())   #enc_output (24256, 32), enc_slf(97024, 10, 10)
#
#         # enc_output = torch.reshape(enc_output, (batch, stock, -1))  #(32, 758, 32)
#         # HGAT_output = self.HGAT(enc_output, H, adj, n_hid)   #here in HGAT_out (32,758,8) , stock embedding update to 8
#         # HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))   #(24256, 8)
#         # seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale
#         seq_logit = self.classifier(enc_output)
#         return seq_logit  #(24256, 3)

# class Bern_HGTAN(nn.Module):
#     ''' A sequence to sequence model with attention mechanism. '''
#     def __init__(
#             self,
#             rnn_unit, n_hid, n_class,
#             feature,
#             d_word_vec, d_model,
#             n_head, d_k, d_v, dropout,
#             tgt_emb_prj_weight_sharing):
#         super().__init__()
#         self.linear = nn.Linear(feature, d_word_vec)
#         self.rnn = nn.GRU(d_word_vec,
#                           rnn_unit,
#                           num_layers=2,
#                           batch_first=True,
#                           bidirectional=False)
#         self.gcn = GCN(feature, n_hid, n_class, dropout)
#         self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)   #for get gru
#
#         self.HGAT = HGAT(nfeat=rnn_unit,
#                          nhid=n_hid,
#                          dropout=dropout,
#                          )
#
#         self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
#         nn.init.xavier_normal_(self.tgt_word_prj.weight)
#
#         assert d_model == d_word_vec, \
#         'To facilitate the residual connections, \
#          the dimensions of all module outputs shall be the same.'
#
#         if tgt_emb_prj_weight_sharing:
#             # Share the weight matrix between target word embedding & the final logit dense layer
#             self.x_logit_scale = (d_model ** -0.5)
#         else:
#             self.x_logit_scale = 1.
#
#     def forward(self, src_seq, H, adj, n_hid):
#         src_seq = self.linear(src_seq)
#         batch = src_seq.size(0)
#         stock = src_seq.size(1)
#         seq_len = src_seq.size(2)
#         dim = src_seq.size(3)
#         src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim))
#
#         slf_attn_mask = get_subsequent_mask(src_seq).bool()
#         gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
#         rnn_output, *_ = self.rnn(src_seq)   #output, hidden state
#         enc_output, enc_slf_attn = self.temp_attn(
#             rnn_output, rnn_output, rnn_output, mask=slf_attn_mask.bool())
#
#         enc_output = torch.reshape(enc_output, (batch, stock, -1))
#
#         HGAT_output = self.HGAT(enc_output, H, adj, n_hid)   #here in HGAT
#         HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))
#         seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale
#
#         return seq_logit



# class G_T(nn.Module):
#     ''' A sequence to sequence model with attention mechanism. '''
#     def __init__(
#             self,
#             rnn_unit, n_hid, n_class,
#             feature,
#             d_word_vec, d_model,
#             n_head, d_k, d_v, dropout,
#             tgt_emb_prj_weight_sharing):
#         super().__init__()
#         self.linear = nn.Linear(feature, d_word_vec)
#         self.m = nn.Linear(rnn_unit, n_hid)
#         self.rnn = nn.GRU(d_word_vec,
#                           rnn_unit,
#                           num_layers=2,
#                           batch_first=True,
#                           bidirectional=False)
#         self.gcn = GCN(feature, n_hid, n_class, dropout)
#         self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)   #for get gru
#         self.HGAT = HGAT(nfeat=rnn_unit,
#                          nhid=n_hid,
#                          dropout=dropout,
#                          )
#
#         self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
#         nn.init.xavier_normal_(self.tgt_word_prj.weight)
#
#         assert d_model == d_word_vec, \
#         'To facilitate the residual connections, \
#          the dimensions of all module outputs shall be the same.'
#
#         if tgt_emb_prj_weight_sharing:
#             # Share the weight matrix between target word embedding & the final logit dense layer
#             self.x_logit_scale = (d_model ** -0.5)
#         else:
#             self.x_logit_scale = 1.
#
#     def forward(self, src_seq, H, adj, n_hid):   #src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
#         new_src = []
#         for i in range(src_seq.shape[0]):
#             x0, x_ = src_seq[i,:,:,:], []   #x0 (758,10,10)
#             for i in range(x0.shape[1]):
#                 x1 = x0[:,i,:]   #(758, 10)
#                 x2 = self.gcn(x1, adj) #x2(758, 10)
#                 x_.append(x2)
#                 y = torch.stack(x_, dim=1) #y(758, 10, 10)
#             new_src.append(y)
#             z = torch.stack(new_src, dim=0) #stack input tensors  (32, 758, 10,10)
#             #new_src.append(x_)
#         #new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
#         #new_src=torch.FloatTensor(new_src)
#         src_seq = self.linear(z)   #z(32, 758, 10,10)
#         #src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
#         batch = src_seq.size(0)
#         stock = src_seq.size(1)
#         seq_len = src_seq.size(2)
#         dim = src_seq.size(3)
#         src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim)) #(24256, 10, 16)
#
#         slf_attn_mask = get_subsequent_mask(src_seq).bool()
#         #gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
#         rnn_output, *_ = self.rnn(src_seq)   #output, hidden state      rnn_out tensor (24256, 10, 32)
#
#         enc_output, enc_slf_attn = self.temp_attn( rnn_output, rnn_output, rnn_output, mask=slf_attn_mask.bool())   #enc_output (24256, 32), enc_slf(97024, 10, 10)
#
#         #enc_output = torch.reshape(enc_output, (batch, stock, -1))  #(32, 758, 32)
#         #HGAT_output = self.HGAT(enc_output, H, adj, n_hid)   #here in HGAT_out (32,758,8) , stock embedding update to 8
#         #HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))   #(24256, 8)
#
#         ###m = nn.Linear(src_seq.shape[0], n_hid) #if directly use number it will be on cpu
#         HGAT_output = self.m(enc_output)
#         seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale
#         return seq_logit  #(24256, 3)
#
#
#
# class G_N(nn.Module):
#     ''' A sequence to sequence model with attention mechanism. '''
#     def __init__(
#             self,
#             rnn_unit, n_hid, n_class,
#             feature,
#             d_word_vec, d_model,
#             n_head, d_k, d_v, dropout,
#             tgt_emb_prj_weight_sharing):
#         super().__init__()
#         self.linear = nn.Linear(feature, d_word_vec)
#         self.m = nn.Linear(rnn_unit, n_hid)
#         self.rnn = nn.GRU(d_word_vec,
#                           rnn_unit,
#                           num_layers=2,
#                           batch_first=True,
#                           bidirectional=False)
#         self.gcn = GCN(feature, n_hid, n_class, dropout)
#         self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)   #for get gru
#         self.HGAT = HGAT(nfeat=rnn_unit,
#                          nhid=n_hid,
#                          dropout=dropout,
#                          )
#
#         self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
#         nn.init.xavier_normal_(self.tgt_word_prj.weight)
#
#         assert d_model == d_word_vec, \
#         'To facilitate the residual connections, \
#          the dimensions of all module outputs shall be the same.'
#
#         if tgt_emb_prj_weight_sharing:
#             # Share the weight matrix between target word embedding & the final logit dense layer
#             self.x_logit_scale = (d_model ** -0.5)
#         else:
#             self.x_logit_scale = 1.
#
#     def forward(self, src_seq, H, adj, n_hid):   #src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
#         new_src = []
#         for i in range(src_seq.shape[0]):
#             x0, x_ = src_seq[i,:,:,:], []   #x0 (758,10,10)
#             for i in range(x0.shape[1]):
#                 x1 = x0[:,i,:]   #(758, 10)
#                 x2 = self.gcn(x1, adj) #x2(758, 10)
#                 x_.append(x2)
#                 y = torch.stack(x_, dim=1) #y(758, 10, 10)
#             new_src.append(y)
#             z = torch.stack(new_src, dim=0) #stack input tensors  (32, 758, 10,10)
#             #new_src.append(x_)
#         #new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
#         #new_src=torch.FloatTensor(new_src)
#         src_seq = self.linear(z)   #z(32, 758, 10,10)
#         #src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
#         batch = src_seq.size(0)
#         stock = src_seq.size(1)
#         seq_len = src_seq.size(2)
#         dim = src_seq.size(3)
#         src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim)) #(24256, 10, 16)
#
#         slf_attn_mask = get_subsequent_mask(src_seq).bool()
#         #gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
#         rnn_output, *_ = self.rnn(src_seq)   #output, hidden state      rnn_out tensor (24256, 10, 32)
#
#         #enc_output, enc_slf_attn = self.temp_attn( rnn_output, rnn_output, rnn_output, mask=slf_attn_mask.bool())   #enc_output (24256, 32), enc_slf(97024, 10, 10)
#
#         enc_output = rnn_output[:, -1, :]
#         enc_output = torch.reshape(enc_output, (batch, stock, -1))  #(32, 758, 32)
#         HGAT_output = self.HGAT(enc_output, H, adj, n_hid)   #here in HGAT_out (32,758,8) , stock embedding update to 8
#         HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))   #(24256, 8)
#
#         ###m = nn.Linear(src_seq.shape[0], n_hid) #if directly use number it will be on cpu
#         #HGAT_output = self.m(enc_output)
#         seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale
#         return seq_logit  #(24256, 3)
#
#
# class END_GCN_HGTAN(nn.Module):  #not good no need add residual
#     ''' A sequence to sequence model with attention mechanism. '''
#     def __init__(
#             self,
#             rnn_unit, n_hid, n_class,
#             feature,
#             d_word_vec, d_model,
#             n_head, d_k, d_v, dropout,
#             tgt_emb_prj_weight_sharing):
#         super().__init__()
#         self.linear = nn.Linear(feature, d_word_vec)
#         self.rnn = nn.GRU(d_word_vec,
#                           rnn_unit,
#                           num_layers=2,
#                           batch_first=True,
#                           bidirectional=False)
#         self.gcn = GCN(feature, n_hid, n_class, dropout)
#
#         self.enc = EncoderRNN(d_word_vec, rnn_unit)
#         self.attn = Attn('concat', rnn_unit)
#         self.dec = LuongAttnDecoderRNN('concat', d_word_vec, rnn_unit, output_size=32) #?output size not konw
#
#         self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)   #for get gru
#         self.HGAT = HGAT(nfeat=rnn_unit,
#                          nhid=n_hid,
#                          dropout=dropout,
#                          )
#
#         self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
#         nn.init.xavier_normal_(self.tgt_word_prj.weight)
#
#         assert d_model == d_word_vec, \
#         'To facilitate the residual connections, \
#          the dimensions of all module outputs shall be the same.'
#
#         if tgt_emb_prj_weight_sharing:
#             # Share the weight matrix between target word embedding & the final logit dense layer
#             self.x_logit_scale = (d_model ** -0.5)
#         else:
#             self.x_logit_scale = 1.
#
#     def forward(self, src_seq, H, adj, n_hid):   #src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
#         new_src = []
#         for i in range(src_seq.shape[0]):
#             x0, x_ = src_seq[i,:,:,:], []   #x0 (758,10,10)
#             for i in range(x0.shape[1]):
#                 x1 = x0[:,i,:]   #(758, 10)
#                 x2 = self.gcn(x1, adj) #x2(758, 10)
#                 x_.append(x2)
#                 y = torch.stack(x_, dim=1) #y(758, 10, 10)
#             new_src.append(y)
#             z = torch.stack(new_src, dim=0) #stack input tensors  (32, 758, 10,10)
#             #new_src.append(x_)
#         #new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
#         #new_src=torch.FloatTensor(new_src)
#         src_seq = self.linear(z)   #z(32, 758, 10,10)
#         #src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
#         batch = src_seq.size(0)
#         stock = src_seq.size(1)
#         seq_len = src_seq.size(2)
#         dim = src_seq.size(3)
#         x = src_seq
#
#
#         src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim)) #(24256, 10, 16)
#
#         slf_attn_mask = get_subsequent_mask(src_seq).bool()
#         #gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
#         rnn_output, *_ = self.rnn(src_seq)   #output, hidden state      rnn_out tensor (24256, 10, 32)
#
#         # out1, hid1 = self.enc(src_seq, src_seq.size(2))   #comment as no need pad or unpad
#         out2, hid2 = self.dec(src_seq, 32, rnn_output)  # 32 is rnn_unit, the hid, x is the 4 dim src_seq (32, 758, 10, 16)
#         #out2(24256,11,32)
#         #rnn_output += out2  #in pytorch should not wtire like this, otherwise "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation, which is output 0 of ViewBackward, is at version 1; expected version 0 instead.
# #Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True)" just normal
#         rnn_output = rnn_output + out2
#
#         enc_output, enc_slf_attn = self.temp_attn(
#             rnn_output, rnn_output, rnn_output, mask=slf_attn_mask.bool())   #enc_output (24256, 32), enc_slf(97024, 10, 10)
#
#         enc_output = torch.reshape(enc_output, (batch, stock, -1))  #(32, 758, 32)
#
#         HGAT_output = self.HGAT(enc_output, H, adj, n_hid)   #here in HGAT_out (32,758,8) , stock embedding update to 8
#         HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))   #(24256, 8)
#         seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale
#
#         return seq_logit  #(24256, 3)
#
#
# class DE_GCN_HGTAN(nn.Module):   #not good
#     ''' A sequence to sequence model with attention mechanism. '''
#     def __init__(
#             self,
#             rnn_unit, n_hid, n_class,
#             feature,
#             d_word_vec, d_model,
#             n_head, d_k, d_v, dropout,
#             tgt_emb_prj_weight_sharing):
#         super().__init__()
#         self.linear = nn.Linear(feature, d_word_vec)
#         self.rnn = nn.GRU(d_word_vec,
#                           rnn_unit,
#                           num_layers=2,
#                           batch_first=True,
#                           bidirectional=False)
#         self.gcn = GCN(feature, n_hid, n_class, dropout)
#         self.enc = EncoderRNN(d_word_vec, rnn_unit)
#         self.attn = Attn('concat', rnn_unit)
#         self.dec = LuongAttnDecoderRNN('concat', d_word_vec, rnn_unit, output_size=32) #?output size not konw
#
#         self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)   #for get gru
#         self.HGAT = HGAT(nfeat=rnn_unit,
#                          nhid=n_hid,
#                          dropout=dropout,
#                          )
#
#         self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
#         nn.init.xavier_normal_(self.tgt_word_prj.weight)
#
#         assert d_model == d_word_vec, \
#         'To facilitate the residual connections, \
#          the dimensions of all module outputs shall be the same.'
#
#         if tgt_emb_prj_weight_sharing:
#             # Share the weight matrix between target word embedding & the final logit dense layer
#             self.x_logit_scale = (d_model ** -0.5)
#         else:
#             self.x_logit_scale = 1.
#
#     def forward(self, src_seq, H, adj, n_hid):   #src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
#         new_src = []
#         for i in range(src_seq.shape[0]):
#             x0, x_ = src_seq[i,:,:,:], []   #x0 (758,10,10)
#             for i in range(x0.shape[1]):
#                 x1 = x0[:,i,:]   #(758, 10)
#                 x2 = self.gcn(x1, adj) #x2(758, 10)
#                 x_.append(x2)
#                 y = torch.stack(x_, dim=1) #y(758, 10, 10)
#             new_src.append(y)
#             z = torch.stack(new_src, dim=0) #stack input tensors  (32, 758, 10,10)
#         src_seq = self.linear(z)   #z(32, 758, 10,10)
#         #src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
#         batch = src_seq.size(0)
#         stock = src_seq.size(1)
#         seq_len = src_seq.size(2)
#         dim = src_seq.size(3)
#         x = src_seq
#
#         src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim)) #(24256, 10, 16)
#         slf_attn_mask = get_subsequent_mask(src_seq).bool()
#         rnn_output, *_ = self.rnn(src_seq)   #output, hidden state      rnn_out tensor (24256, 10, 32)
#
#         out2, hid2 = self.dec(src_seq, 32, rnn_output)  # 32 is rnn_unit, the hid, x is the 4 dim src_seq (32, 758, 10, 16)
#         #out2(24256,11,32)
#         #rnn_output += out2  #in pytorch should not wtire like this, otherwise "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation, which is output 0 of ViewBackward, is at version 1; expected version 0 instead.
# #Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True)" just normal
#         #rnn_output = rnn_output + out2
#
#         enc_output, enc_slf_attn = self.temp_attn(
#             out2, out2, out2, mask=slf_attn_mask.bool())  #change from rnnoutput  #enc_output (24256, 32), enc_slf(97024, 10, 10)
#
#         enc_output = torch.reshape(enc_output, (batch, stock, -1))  #(32, 758, 32)
#
#         HGAT_output = self.HGAT(enc_output, H, adj, n_hid)   #here in HGAT_out (32,758,8) , stock embedding update to 8
#         HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))   #(24256, 8)
#         seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale
#
#         return seq_logit  #(24256, 3)



# class Notemp_HGTAN(nn.Module):
#     ''' A sequence to sequence model with attention mechanism. '''
#     def __init__(
#             self,
#             rnn_unit, n_hid, n_class,
#             feature,
#             d_word_vec, d_model,
#             n_head, d_k, d_v, dropout,
#             tgt_emb_prj_weight_sharing):
#         super().__init__()
#         self.linear = nn.Linear(feature, d_word_vec)
#         self.rnn = nn.GRU(d_word_vec,
#                           rnn_unit,
#                           num_layers=2,
#                           batch_first=True,
#                           bidirectional=False)
#         self.gcn = GCN(feature, n_hid, n_class, dropout)
#         self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)   #for get gru
#         self.HGAT = HGAT1(nfeat=rnn_unit,
#                          nhid=n_hid,
#                          dropout=dropout,
#                          )
#
#         self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
#         nn.init.xavier_normal_(self.tgt_word_prj.weight)
#
#         assert d_model == d_word_vec, \
#         'To facilitate the residual connections, \
#          the dimensions of all module outputs shall be the same.'
#
#         if tgt_emb_prj_weight_sharing:
#             # Share the weight matrix between target word embedding & the final logit dense layer
#             self.x_logit_scale = (d_model ** -0.5)
#         else:
#             self.x_logit_scale = 1.
#
#     def forward(self, src_seq, H, adj, n_hid):   #src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
#         new_src = []
#         for i in range(src_seq.shape[0]):
#             x0, x_ = src_seq[i,:,:,:], []   #x0 (758,10,10)
#             for i in range(x0.shape[1]):
#                 x1 = x0[:,i,:]   #(758, 10)
#                 x2 = self.gcn(x1, adj) #x2(758, 10)
#                 x_.append(x2)
#                 y = torch.stack(x_, dim=1) #y(758, 10, 10)
#             new_src.append(y)
#             z = torch.stack(new_src, dim=0) #stack input tensors  (32, 758, 10,10)
#             #new_src.append(x_)
#         #new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
#         #new_src=torch.FloatTensor(new_src)
#         src_seq = self.linear(z)   #z(32, 758, 10,10)
#         #src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
#         batch = src_seq.size(0)
#         stock = src_seq.size(1)
#         seq_len = src_seq.size(2)
#         dim = src_seq.size(3)
#         src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim)) #(24256, 10, 16)
#
#         slf_attn_mask = get_subsequent_mask(src_seq).bool()
#         #gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
#         rnn_output, *_ = self.rnn(src_seq)   #output, hidden state      rnn_out tensor (24256, 10, 32)
#         #enc_output, enc_slf_attn = self.temp_attn(
#             #rnn_output, rnn_output, rnn_output, mask=slf_attn_mask.bool())   #enc_output (24256, 32), enc_slf(97024, 10, 10)
#
#         enc_output = rnn_output[:, -1, :]
#         enc_output = torch.reshape(enc_output, (batch, stock, -1))  #(32, 758, 32)
#
#
#
#         HGAT_output = self.HGAT(enc_output, H, adj, n_hid)   #here in HGAT_out (32,758,8) , stock embedding update to 8
#         HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))   #(24256, 8)
#         seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale
#
#         return seq_logit  #(24256, 3)
#
#
# class MLP_HGTAN(nn.Module):
#     ''' A sequence to sequence model with attention mechanism. '''
#     def __init__(
#             self,
#             rnn_unit, n_hid, n_class,
#             feature,
#             d_word_vec, d_model,
#             n_head, d_k, d_v, dropout,
#             tgt_emb_prj_weight_sharing):
#         super().__init__()
#         self.linear = nn.Linear(feature, d_word_vec)
#         self.m = nn.Linear(rnn_unit, n_hid)
#         self.rnn = nn.GRU(d_word_vec,
#                           rnn_unit,
#                           num_layers=2,
#                           batch_first=True,
#                           bidirectional=False)
#         self.gcn = GCN(feature, n_hid, n_class, dropout)
#         self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)   #for get gru
#         self.HGAT = HGAT(nfeat=rnn_unit,
#                          nhid=n_hid,
#                          dropout=dropout,
#                          )
#
#         self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
#         nn.init.xavier_normal_(self.tgt_word_prj.weight)
#
#         assert d_model == d_word_vec, \
#         'To facilitate the residual connections, \
#          the dimensions of all module outputs shall be the same.'
#
#         if tgt_emb_prj_weight_sharing:
#             # Share the weight matrix between target word embedding & the final logit dense layer
#             self.x_logit_scale = (d_model ** -0.5)
#         else:
#             self.x_logit_scale = 1.
#
#     def forward(self, src_seq, H, adj, n_hid):   #src_seq(32,758, 10, 10)  adj(758,758), H(758, 62), n_hid(8)
#         new_src = []
#         for i in range(src_seq.shape[0]):
#             x0, x_ = src_seq[i,:,:,:], []   #x0 (758,10,10)
#             for i in range(x0.shape[1]):
#                 x1 = x0[:,i,:]   #(758, 10)
#                 x2 = self.gcn(x1, adj) #x2(758, 10)
#                 x_.append(x2)
#                 y = torch.stack(x_, dim=1) #y(758, 10, 10)
#             new_src.append(y)
#             z = torch.stack(new_src, dim=0) #stack input tensors  (32, 758, 10,10)
#             #new_src.append(x_)
#         #new_src = torch.tensor([item.cpu().detach().numpy() for item in new_src]).cuda()   # https://blog.csdn.net/qq_27261889/article/details/100483097
#         #new_src=torch.FloatTensor(new_src)
#         src_seq = self.linear(z)   #z(32, 758, 10,10)
#         #src_seq = self.linear(src_seq)       #src_seq(32,758, 10, 16)
#         batch = src_seq.size(0)
#         stock = src_seq.size(1)
#         seq_len = src_seq.size(2)
#         dim = src_seq.size(3)
#         src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim)) #(24256, 10, 16)
#
#         slf_attn_mask = get_subsequent_mask(src_seq).bool()
#         #gcn_output = self.gcn(src_seq)   #src_seq(24256,10, 16)
#         rnn_output, *_ = self.rnn(src_seq)   #output, hidden state      rnn_out tensor (24256, 10, 32)
#
#         #enc_output, enc_slf_attn = self.temp_attn( rnn_output, rnn_output, rnn_output, mask=slf_attn_mask.bool())   #enc_output (24256, 32), enc_slf(97024, 10, 10)
#         enc_output = rnn_output[:, -1, :] #should be  (24256, 32)
#
#         #enc_output = torch.reshape(enc_output, (batch, stock, -1))  #(32, 758, 32)
#         #HGAT_output = self.HGAT(enc_output, H, adj, n_hid)   #here in HGAT_out (32,758,8) , stock embedding update to 8
#         #HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))   #(24256, 8)
#
#         ###m = nn.Linear(src_seq.shape[0], n_hid) #if directly use number it will be on cpu
#         HGAT_output = self.m(enc_output)
#         seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale
#         return seq_logit  #(24256, 3)



