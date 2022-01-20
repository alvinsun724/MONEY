import torch
import torch.nn.functional as F
import time
from sklearn import metrics
import torch.utils.data as Data
from load_data import load_EOD_data, get_batch, get_fund_adj_H, get_industry_adj
import copy
import argparse
import numpy as np
from models import DGCN_HGN_AD  #Notemp_HGTAN, MLP_HGTAN, G_T, G_N,  Bern_HGTAN, Bern_HGN_No_Att,
from Optim import ScheduledOptim
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('-length', default=10,
                    help='length of historical sequence for feature')    #change to 20 #5, 10, 20 historical price
parser.add_argument('-train_index', type=int, default=1021)  # 0.6
parser.add_argument('-valid_index', type=int, default=1361)  # 0.2
parser.add_argument('-feature', default=10, help='input_size')
parser.add_argument('-n_class', default=3, help='output_size')
parser.add_argument('-epoch', type=int, default=600)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('--rnn_unit', type=int, default=32, help='Number of hidden units.')
parser.add_argument('-d_model', type=int, default=16)
parser.add_argument('-d_k', type=int, default=8)
parser.add_argument('-d_v', type=int, default=8)
parser.add_argument('-n_head', type=int, default=4)
parser.add_argument('-n_layers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-proj_share_weight', default='True')
parser.add_argument('-log', default='../10_days/lstm+trans+HGAT3_5_valid1')
parser.add_argument('-save_model', default='../10_days/lstm+HGAT3_5_valid1')
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-label_smoothing', default='False') #change to false
parser.add_argument('-n_warmup_steps', type=int, default=4000)
parser.add_argument('-steps', default=1, help='steps to make prediction')
parser.add_argument('-beta', type=float, default=1e-2, help='beta for adversial loss') # https://github.com/yuxiangalvin/Stock-Move-Prediction-with-Adversarial-Training-Replicate

args = parser.parse_args()
args.cuda = not args.no_cuda
args.d_word_vec = args.d_model  #dim of embedding, price embed into such dimensional space

def prepare_dataloaders(eod_data, gt_data, args):
    # ========= Preparing DataLoader =========#     eod_data ndarray(758,1702,10), gt_data(758, 1702)
    EOD, GT = [], []
    for i in range(eod_data.shape[1] - args.length):
        eod, gt = get_batch(eod_data, gt_data, i, args.length)
        EOD.append(eod)   #eod(758, 10, 10)
        GT.append(gt)

    train_eod, train_gt = EOD[:args.train_index], GT[:args.train_index]
    valid_eod, valid_gt = EOD[args.train_index:args.valid_index], GT[args.train_index:args.valid_index]
    test_eod, test_gt = EOD[args.valid_index:], GT[args.valid_index:]

    #train_eod, valid_eod, test_eod = torch.FloatTensor(train_eod), torch.FloatTensor(valid_eod), torch.FloatTensor(test_eod) #train_eod(1021,758,10,10) Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor

    train_eod, valid_eod, test_eod = torch.FloatTensor(np.array(train_eod)), torch.FloatTensor(np.array(valid_eod)), torch.FloatTensor(np.array(test_eod))
    train_gt, valid_gt, test_gt = torch.LongTensor(np.array(train_gt)), torch.LongTensor(np.array(valid_gt)), torch.LongTensor(np.array(test_gt))

    train_dataset = Data.TensorDataset(train_eod, train_gt)
    valid_dataset = Data.TensorDataset(valid_eod, valid_gt)
    test_dataset = Data.TensorDataset(test_eod, test_gt)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, drop_last=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, drop_last=True)
    return train_loader, valid_loader, test_loader

eod_data, ground_truth = load_EOD_data() #eod(758,1702,10)  truth (758,1702)
train_loader, valid_loader, test_loader = prepare_dataloaders(eod_data, ground_truth, args)

fund_adj_tensor_H = get_fund_adj_H()  #ndarray(28, 758, 62)
adj = torch.Tensor(get_industry_adj()) #(758, 758)
fund_adj_tensor_H = torch.Tensor(fund_adj_tensor_H) # fund_adj_tensor_H (28,758,62)

Htensor2 = torch.randn(0)
for i in range(28): # total 28 fund files
    fund = fund_adj_tensor_H[i]

for i in range(28):
    Htensor = torch.randn(0)  #(61, 758, 62)
    for j in range(61):
        Htensor = torch.cat([Htensor, torch.Tensor(fund_adj_tensor_H[i]).unsqueeze(0)], dim=0)  #(61, 758, 62)
    Htensor2 = torch.cat([Htensor2, torch.Tensor(Htensor)], dim=0)   #Ht2(1708, 758, 62)  61*28, so 1021 train, 0-16 train, 16-22 val, 22-28 train


def cal_performance(pred, pred_adv, gold, smoothing=False): #add pre_adv
    loss = cal_loss(pred, pred_adv, gold, smoothing) #add pre_adv

    pred = pred.max(1)[1]# + pred_adv.max(1)[1] #add pred_adv.max(1)[1] make it more ?????
    gold = gold.contiguous().view(-1)

    percision = metrics.precision_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='macro')
    recall = metrics.recall_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='macro')
    f1_score = metrics.f1_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='weighted')

    n_correct = pred.eq(gold)
    n_correct = n_correct.sum().item()

    return loss, n_correct, percision, recall, f1_score

def cal_loss(pred, pred_adv, gold, smoothing):  #gold(32, 758)， pred(24256, 3)
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)   #(24256) variable has to be contiguous before use view make it whole piece, -1 means one dim
    if smoothing:
        eps = 0.1 #maybe improve a little as training accuracy is highly better than test
        n_class = 3  #gold.view(-1,1) (24256,1) which is tensor([[1],[2],[2]]) shows the corresponding label
        #target.scatter(dim, index, src) put src value according to dim 1 refers to 3 in (24256,3) according to index in target
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)   #one_hot(24256, 3), pred(24256,3) gold (24256), 1 dim, one_hot([[0,1,0], [0,0,1]])
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1) #still (24256, 3)  one_hot([[0.05,0.9,0.05],[0.05,0.05,0.9],[0.05,0.05,0.9]])
        log_prb = F.log_softmax(pred, dim=1)   #(24256,3)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, weight=torch.tensor([1.5, 1.0, 1.0]).cuda()) + args.beta * F.cross_entropy(pred_adv, gold, weight=torch.tensor([1.5, 1.0, 1.0]).cuda()) #delete the reduction=sum as not work
    return loss


device = torch.device('cuda' if args.cuda else 'cpu')
model = DGCN_HGN_AD(rnn_unit=args.rnn_unit,n_hid=args.hidden,n_class=args.n_class, feature=args.feature,
                 tgt_emb_prj_weight_sharing=args.proj_share_weight,d_k=args.d_k,d_v=args.d_v, d_model=args.d_model,
        d_word_vec=args.d_word_vec, n_head=args.n_head, dropout=args.dropout).to(device)  #change   GCN_HGN is not bad only F1 down 0.2%

optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09), args.d_model, args.n_warmup_steps)

def try_train():
    total_loss = 0
    total_accu = 0
    total_f1 = 0
    n_count = 0
    model.train()
    for step, (eod, gt) in enumerate(train_loader):
        #with autograd.detect_anomaly():#Htensor2 (1708,758,62)
        H = Htensor2[args.batch_size*step] #Htensor2[0][32][64][96]...[1021] which step also reach 32 nearly, for train_idx
        Eod, Gt, H_,adj_= eod.to(device), gt.to(device), H.to(device), adj.to(device)

        # forward
        optimizer.zero_grad()
        pred, pred_adv = model(Eod,H_,adj_,args.hidden) #Eod price (32,758, 10, 10), adj(758,758), H(758, 62), n_hid(8)  #add pred_adv
        # backward
        loss, n_correct, percision, recall, f1_score = cal_performance(pred, pred_adv, Gt, smoothing=args.label_smoothing) #add pred_adv
        #should add weighted
        loss.backward()

        optimizer.step_and_update_lr()

        total_loss += loss.item()
        total_accu += n_correct
        total_f1 += f1_score  #all total_f1 are newly added
        n_count += Eod.size(0) * Eod.size(1)

    epoch_loss = total_loss / n_count
    accuracy = total_accu / n_count

    print(
        ' - (Training) loss:{loss:8.5f}, accuracy:{accu:3.3f}%,  percision:{perc:3.3f}%,  recall:{recall:3.3f}%,  f1_score:{f1:3.3f}% , ' \
        'elapse: {elapse:3.3f} min'.format(
            loss=epoch_loss, accu=100 * accuracy,
            perc=100 * percision, recall=100 * recall, f1=100 * f1_score,
            elapse=(time.time() - start) / 60))

def test():
    total_loss = 0
    total_accu = 0
    total_f1 = 0
    total_val_f1 = 0
    total_test_f1 = 0
    n_count = 0
    valid_pred = []
    test_pred = []
    test_pred_before = []
    model.eval()
    with torch.no_grad():
        for step, (eod, gt) in enumerate(valid_loader):
            H = Htensor2[args.train_index+args.batch_size * step]
            # prepare data
            Eod, Gt, H_, adj_,= eod.to(device), gt.to(device), H.to(device), adj.to(device)

            # forward
            pred, pred_adv = model(Eod, H_, adj_, args.hidden)  #add pred_adv
            loss, n_correct, val_percision, val_recall, val_f1_score = cal_performance(pred,pred_adv, Gt, smoothing=False) #add pred_adv

            pred = pred.max(1)[1]   #feel useless
            pred = pred.cuda().data.cpu().numpy()   #feel useless
            valid_pred.extend(pred)       #feel useless

            total_loss += loss.item()
            total_accu += n_correct

            total_val_f1 += val_f1_score

            n_count += Eod.size(0) * Eod.size(1)

        for step, (eod, gt) in enumerate(test_loader):
            H = Htensor2[args.valid_index + args.batch_size * step]
            Eod, Gt, H_, adj_, = eod.to(device), gt.to(device), H.to(device), adj.to(device)
            pred, pred_adv = model(Eod, H_, adj_, args.hidden)   #add pred_adv
            loss, n_correct, test_percision, test_recall, test_f1_score = cal_performance(pred, pred_adv, Gt, smoothing=False)  #add pred_adv


            pred = F.softmax(pred, dim=1)
            pred = pred.cuda().data.cpu().numpy()
            test_pred_before.append(pred)
            pred = torch.Tensor(pred).to(device)
            pred = pred.max(1)[1]

            test_pred.extend(pred)     #feel useless

            total_loss += loss.item()
            total_accu += n_correct

            total_test_f1 += test_f1_score

            n_count += Eod.size(0) * Eod.size(1)

    test_epoch_loss = total_loss / n_count
    test_accuracy = total_accu / n_count
    #f1 = total_f1 / n_count

    val_epoch_loss = total_loss / n_count
    val_accuracy = total_accu / n_count

    val_f1 = total_val_f1  / n_count

    test_f1 = total_test_f1 / n_count
    return val_accuracy,  test_accuracy, val_f1, test_f1
    #return val_accuracy,  test_accuracy

for epoch_i in range(args.epoch):
    print('[ Epoch', epoch_i, ']')

    start = time.time()
    try_train()

    min_loss_val = 10
    min_epoch = 150
    best_model = None
    best_accu = 0
    best_val_acc = 0
    best_f1 = 0
    test_f1 = 0

    tmp_test_acc, val_acc, val_f1, tmp_test_f1 = test()    #tmp_test_acc, val_acc = test()

    if (val_acc > best_val_acc and epoch_i>min_epoch) and (val_f1 > best_f1):  #and best_val_acc>0.381   epoch_i > min_epoch
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        best_f1 = val_f1
        test_f1 = tmp_test_f1

    # if valid_accu >= max(valid_accus) and epoch_i > min_epoch:
        best_model = copy.deepcopy(model)
        print("save best model")
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': args,
            'epoch': epoch_i}

        model_name = '10_weighted' +'DGCN_HGN_AD'  +'_accu_{accu:3.3f}.chkpt'.format(accu=100 * best_val_acc)   #no_hyperedge_fts_i'
        torch.save(checkpoint, model_name)



