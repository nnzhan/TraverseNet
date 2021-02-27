import torch
from utils import metrics
import numpy as np
import time
class TGtrainer:
    #to do: to be deleted from project
    def __init__(self, model, optimizer, lr_scheduler, loss, dataloader, params, seq_out_len, scaler, device):
        self.model = model
        self.model.to(device)
        self.dataloader = dataloader
        self.scaler = scaler
        self.device = device

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss
        self.clip = params['clip']
        self.print_every = params['print_every']
        self.seq_out_len = seq_out_len
        self.batches_seen = 0

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
            metrics = self.train(trainx, trainy, self.batches_seen)

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % self.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
            self.batches_seen +=1
        self.lr_scheduler.step()

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
                metrics = self.eval(testx, testy)

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        t2 = time.time()

        return np.mean(valid_loss),np.mean(valid_mape),np.mean(valid_rmse), t2-t1


    def train(self, src, tgt, batches_seen):
        self.model.train()
        self.optimizer.zero_grad()
        src_x = {'v': src}
        tgt_x = {'v': tgt}
        dummy = torch.zeros(10).requires_grad_()

        predict = self.model(src_x, tgt_x, dummy, batches_seen)
        real = torch.unsqueeze(tgt[:,0,:,:],dim=1)
        real = self.scaler.inverse_transform(real)
        predict = self.scaler.inverse_transform(predict)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        # mae = util.masked_mae(predict,real,0.0).item()
        mape = metrics.masked_mape(predict,real).item()
        rmse = metrics.masked_rmse(predict,real).item()
        return loss.item(),mape,rmse


    def eval(self, src, tgt):
        self.model.eval()
        src_x = {'v': src}
        tgt_x = {'v': tgt}
        dummy = torch.zeros(10).requires_grad_()
        predict = self.model(src_x, tgt_x, dummy)
        real = torch.unsqueeze(tgt[:,0,:,:],dim=1)
        real = self.scaler.inverse_transform(real)
        predict = self.scaler.inverse_transform(predict)
        mae, mape, rmse = metrics.metric(predict, real)
        return mae, mape, rmse


    def ev_valid(self, name):
        self.model.eval()
        outputs = []
        realy = []
        dummy = torch.zeros(10).requires_grad_()

        for iter, (x, y) in enumerate(self.dataloader[name+'_loader'].get_iterator()):
            testx = torch.Tensor(x).to(self.device)
            testx = testx.transpose(1, 3)

            testy = torch.Tensor(y).to(self.device)
            testy = testy.transpose(1, 3)[:,:,:,:self.seq_out_len]
            realy.append(testy[:,0,:,:].squeeze())

            src_x = {'v': testx}
            testy = torch.Tensor(y).to(self.device)
            testy = testy.transpose(1, 3)[:,:,:,:self.seq_out_len]
            tgt_x = {'v': testy}
            with torch.no_grad():
                preds = self.model(src_x, tgt_x, dummy)
            outputs.append(preds.squeeze())

        pred = torch.cat(outputs, dim=0)
        realy = torch.cat(realy, dim=0)

        pred = self.scaler.inverse_transform(pred)
        realy = self.scaler.inverse_transform(realy)
        mae, mape, rmse = metrics.metric(pred, realy)
        return mae, mape, rmse

    def ev_test(self,name):
        self.model.eval()
        outputs = []
        realy = []
        # realy = torch.Tensor(self.dataloader['y_'+name]).to(self.device)
        # realy = realy.transpose(1, 3)[:, 0, :, :self.seq_out_len]
        dummy = torch.zeros(10).requires_grad_()

        for iter, (x, y) in enumerate(self.dataloader[name+'_loader'].get_iterator()):
            testx = torch.Tensor(x).to(self.device)
            testx = testx.transpose(1, 3)
            src_x = {'v': testx}

            testy = torch.Tensor(y).to(self.device)
            testy = testy.transpose(1, 3)[:,:,:,:self.seq_out_len]
            realy.append(testy[:,0,:,:].squeeze())

            tgt_x = {'v': testy}
            with torch.no_grad():
                preds = self.model(src_x, tgt_x, dummy)

            outputs.append(preds.squeeze())

        yhat = torch.cat(outputs, dim=0)
        realy = torch.cat(realy, dim=0)

        mae = []
        mape = []
        rmse = []
        for i in range(self.seq_out_len):
            pred = self.scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, :, i]
            real = self.scaler.inverse_transform(real)
            results = metrics.metric(pred, real)
            log = 'Evaluate best model on ' + name + ' data for horizon {:d}, MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
            print(log.format(i + 1, results[0], results[1], results[2]))
            mae.append(results[0])
            mape.append(results[1])
            rmse.append(results[2])
        return mae, mape, rmse
