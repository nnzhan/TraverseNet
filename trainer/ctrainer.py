import torch
from utils import metrics
import numpy as np
import time
import pickle
import dgl
from utils.process import *
class CTrainer:
    # a trainer for models that make predictions once for all future steps.
    def __init__(self, model, optimizer, loss, dataloader, params, seq_out_len, scaler, device):
        self.model = model
        self.model.to(device)
        self.dataloader = dataloader
        self.scaler = scaler
        self.device = device

        self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler
        self.loss = loss
        self.regloss = torch.nn.BCELoss()

        self.clip = params['clip']
        self.print_every = params['print_every']
        self.seq_out_len = seq_out_len
        self.params = params


    def train_epoch(self):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()

        self.dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(self.dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(self.device)
            trainx = trainx.transpose(1, 3)

            trainy = torch.Tensor(y).to(self.device)
            trainy = trainy.transpose(1, 3)[:,:,:,:self.seq_out_len]

            metrics = self.train(trainx, trainy[:, self.params['out_level'], :, :])

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % self.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        return np.mean(train_loss),np.mean(train_mape),np.mean(train_rmse), t2-t1

    def val_epoch(self):
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        t1 = time.time()
        for iter, (x, y) in enumerate(self.dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(self.device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(self.device)
            testy = testy.transpose(1, 3)[:,:,:,:self.seq_out_len]
            with torch.no_grad():
                metrics = self.eval(testx, testy[:, self.params['out_level'],:,:])

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        t2 = time.time()

        return np.mean(valid_loss),np.mean(valid_mape),np.mean(valid_rmse), t2-t1


    def train(self, x, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        dummy = torch.zeros(10).requires_grad_()
        output = self.model(x,dummy)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val,dim=1)
        real = self.scaler.inverse_transform(real,self.params['out_level'])
        predict = self.scaler.inverse_transform(output,self.params['out_level'])
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        #self.lr_scheduler.step()
        # mae = util.masked_mae(predict,real,0.0).item()
        mape = metrics.masked_mape(predict,real).item()
        rmse = metrics.masked_rmse(predict,real).item()
        return loss.item(),mape,rmse


    def eval(self, x, real_val):
        self.model.eval()
        dummy = torch.zeros(10).requires_grad_()
        output = self.model(x,dummy)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val,dim=1)
        real = self.scaler.inverse_transform(real,self.params['out_level'])
        predict = self.scaler.inverse_transform(output,self.params['out_level'])
        loss = self.loss(predict, real)
        mape = metrics.masked_mape(predict,real).item()
        rmse = metrics.masked_rmse(predict,real).item()
        return loss.item(),mape,rmse


    def ev_valid(self,name):
        self.model.eval()
        outputs = []
        realy = []
        dummy = torch.zeros(10).requires_grad_()
        for iter, (x, y) in enumerate(self.dataloader[name+'_loader'].get_iterator()):
            testx = torch.Tensor(x).to(self.device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(self.device)
            testy = testy.transpose(1, 3)[:,:,:,:self.seq_out_len]
            realy.append(testy[:,self.params['out_level'],:,:].squeeze())


            with torch.no_grad():
                preds = self.model(testx,dummy)
                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze())

        yhat = torch.cat(outputs, dim=0)
        realy = torch.cat(realy, dim=0)

        pred = self.scaler.inverse_transform(yhat,self.params['out_level'])
        realy = self.scaler.inverse_transform(realy,self.params['out_level'])
        mae, mape, rmse = metrics.metric(pred, realy)
        return mae, mape, rmse

    def ev_test(self, name):
        self.model.eval()
        outputs = []
        realy = []
        dummy = torch.zeros(10).requires_grad_()

        for iter, (x, y) in enumerate(self.dataloader[name+'_loader'].get_iterator()):
            testx = torch.Tensor(x).to(self.device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(self.device)
            testy = testy.transpose(1, 3)[:,:,:,:self.seq_out_len]
            realy.append(testy[:,self.params['out_level'],:,:].squeeze(dim=1))

            with torch.no_grad():
                preds = self.model(testx,dummy)
                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze(dim=1))

        yhat = torch.cat(outputs, dim=0)
        realy = torch.cat(realy, dim=0)


        mae = []
        mape = []
        rmse = []
        for i in range(self.seq_out_len):
            pred = self.scaler.inverse_transform(yhat[:, :, i],self.params['out_level'])
            real = realy[:, :, i]
            real = self.scaler.inverse_transform(real,self.params['out_level'])
            results = metrics.metric(pred, real)
            log = 'Evaluate best model on ' + name +' data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, results[0], results[1], results[2]))
            mae.append(results[0])
            mape.append(results[1])
            rmse.append(results[2])
        return mae, mape, rmse
