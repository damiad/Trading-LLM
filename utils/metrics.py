import numpy as np
import torch

class Metrics:
    def __init__(self, j):
        self.j = j
        self.rse = []
        self.corr = []
        self.mae = []
        self.mse = []
        self.rmse = []
        self.mape = []
        self.mspe = []
        self.cg0 = []
        self.cgd = []
        self.cgi = []
    def append(self, last_val, pred, true):
        #has to be written in torch not np
        # self.rse.append(RSE(pred, true))
        # self.corr.append(CORR(pred, true))
        # self.mae.append(MAE(pred, true))
        # self.mse.append(MSE(pred, true))
        # self.rmse.append(RMSE(pred, true))
        # self.mape.append(MAPE(pred, true))
        # self.mspe.append(MSPE(pred, true))
        self.cg0.append(CG0(last_val, pred, true))
        self.cgd.append(CGD(last_val, pred, true))
        self.cgi.append(CGI(self.j, last_val, pred, true))
    def compute(self):
        self.rse = np.mean(self.rse)
        self.corr = np.mean(self.corr)
        self.mae = np.mean(self.mae)
        self.mse = np.mean(self.mse)
        self.rmse = np.mean(self.rmse)
        self.mape = np.mean(self.mape)
        self.mspe = np.mean(self.mspe)
        self.cg0 = np.mean(self.cg0)
        self.cgd = np.mean(self.cgd)
        self.cgi = np.mean(self.cgi)



def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

#can be optimised
#not ideal cus we only compare to the first value. 
#maybe should move to np
#0 cuz with respect to the first value
# for now only check the first value
def CG0(last_val, pred, true):
    last_val = last_val.view(-1, 1, pred.size(-1)).expand(-1, pred.size(1), pred.size(2))
    pred_deltas = torch.sign(pred - last_val)
    true_deltas = torch.sign(true - last_val)
    pred_deltas = pred_deltas[:,0,:]
    true_deltas = true_deltas[:,0,:]
    count = torch.sum(pred_deltas == true_deltas)
    # return count.item()
    return count.item() / pred_deltas.size(0) / pred_deltas.size(1)

#now we define accuracy as the % of correctly guessed directions
#D cuz directions
def CGD(last_val, pred, true):
    pred_last = torch.roll(pred, 1, 1)
    true_last = torch.roll(true, 1, 1)

    pred_last[:,0] = last_val
    true_last[:,0] = last_val
    
    pred_deltas = torch.sign(pred - pred_last)
    true_deltas = torch.sign(true - true_last)
    count = torch.sum(pred_deltas == true_deltas)
    # return count.item()
    return count.item() / pred_deltas.size(0) / pred_deltas.size(1)

#next we define accuracy as  
# 1 if min_{k \in i-1,i,i+1,...,i+j}true[k]<= pred[i] <= max_{k \in i-1,i,i+1,...,i+j}true[k] else 0
#  where j is a parameter
# I for interval
def CGI(j, last_val, pred, true):
    if j >= true.size(1):
        return 0
    true_last = torch.cat((last_val.unsqueeze(1), true), 1)
    true_last = true_last.permute(0, 2, 1)

    pred = pred.permute(0, 2, 1)
    pred = pred[:, :, :-j]

    maxpool = torch.nn.MaxPool1d(j+2, stride=1)
    maxvals = maxpool(true_last)
    minvals = -maxpool(-true_last)

    # print("true_last: ", true_last)
    # print("maxvals: ", maxvals)
    # print("minvals: ", minvals)
    # print("pred: ", pred)
    # exit()
    count = torch.sum((pred >= minvals) & (pred <= maxvals))
    # return count.item()
    return count.item() / pred.size(0) / pred.size(2)
    # print("pred_last: " ,pred_last )
    # print("maxpool: ", maxpool(pred_last))



def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
