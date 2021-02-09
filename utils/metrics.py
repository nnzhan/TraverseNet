import torch
import numpy as np


def masked_huber(preds, labels, null_val=0, beta=1):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    assert list(preds.shape)==list(labels.shape), "shapes of two inputs are not equal"
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    se = (preds-labels)**2
    se = se * mask
    se = torch.where(torch.isnan(se), torch.zeros_like(se), se)
    ae = torch.abs(preds-labels)
    ae = ae * mask
    ae = torch.where(torch.isnan(ae), torch.zeros_like(ae), ae)
    loss = torch.where(ae<beta,0.5*se/beta,ae-0.5*beta)
    return torch.mean(loss)


def masked_mse(preds, labels, null_val=0.0001):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels>null_val)
    assert list(preds.shape)==list(labels.shape), "shapes of two inputs are not equal"
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=0.0001):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=0.0001):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels>null_val)
    assert list(preds.shape)==list(labels.shape), "shapes of two inputs are not equal"
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=0.0001):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels>null_val)
    assert list(preds.shape)==list(labels.shape), "shapes of two inputs are not equal"
    mask = mask.float()
    mask /=  torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/torch.abs(labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred,real).item()
    mape = masked_mape(pred,real).item()
    rmse = masked_rmse(pred,real).item()
    return mae,mape,rmse


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()