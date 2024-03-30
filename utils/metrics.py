import numpy as np
import torch


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
#we could also check the delats respectively to the previous value in the array
#maybe should move to np
def CG0_cuda(last_val, pred, true):
    last_val = last_val.view(-1, 1, pred.size(-1)).expand(-1, pred.size(1), pred.size(2))
    pred_deltas = torch.sign(pred - last_val)
    true_deltas = torch.sign(true - last_val)
    count = torch.sum(pred_deltas == true_deltas)
    return count.item()


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
