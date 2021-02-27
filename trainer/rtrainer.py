import time
import numpy as np
import torch
from utils import metrics
import torch.optim as optim


class RTrainer:
    # a trainer for models that make predictions sequentially.
    def __init__(self, model, optimizer, lr_scheduler, loss, dataloader, params, net_params, scaler, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.loss = loss

        self.seq_in_len = net_params['seq_in_len']
        self.seq_out_len = net_params['seq_out_len']
        self.num_nodes = net_params['num_nodes']
        self.in_dim = net_params['in_dim']
        self.out_dim = net_params['out_dim']

        self.clip = params['clip']
        self.print_every = params['print_every']
        self.dataloader = dataloader
        self.device = device
        self.batches_seen = 0
        self.params = params

    def train_epoch(self):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        self.dataloader['train_loader'].shuffle()
        train_iterator = self.dataloader['train_loader'].get_iterator()
        for iter, (x, y) in enumerate(train_iterator):
            x, y = self._prepare_data(x, y)
            metrics = self.train(x,y,self.batches_seen)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % self.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
            self.batches_seen += 1
        self.lr_scheduler.step()

        t2 = time.time()
        return np.mean(train_loss),np.mean(train_mape),np.mean(train_rmse), t2-t1

    def val_epoch(self):
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        t1 = time.time()
        val_iterator = self.dataloader['val_loader'].get_iterator()

        for _, (x, y) in enumerate(val_iterator):
            x, y = self._prepare_data(x, y)
            with torch.no_grad():
                metrics = self.eval(x, y)

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        t2 = time.time()
        return np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse), t2 - t1

    def train(self, x, y, batches_seen):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(x, y, batches_seen)
        if batches_seen == 0:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, eps=1e-3)
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 30, 40, 50], gamma=0.1)
        y = y[...,self.params['out_level']].squeeze()
        y_true = self.scaler.inverse_transform(y, self.params['out_level'])
        y_predicted = self.scaler.inverse_transform(output,self.params['out_level'])
        loss = self.loss(y_predicted, y_true)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape = metrics.masked_mape(y_predicted,y_true).item()
        rmse = metrics.masked_rmse(y_predicted,y_true).item()
        return loss.item(),mape,rmse

    def eval(self, x, y):
        self.model.eval()
        output = self.model(x,y)
        y = y[...,self.params['out_level']].squeeze()
        y_true = self.scaler.inverse_transform(y,self.params['out_level'])
        y_predicted = self.scaler.inverse_transform(output,self.params['out_level'])
        loss = self.loss(y_predicted, y_true)
        mape = metrics.masked_mape(y_predicted,y_true).item()
        rmse = metrics.masked_rmse(y_predicted,y_true).item()
        return loss.item(),mape,rmse


    def ev_valid(self,name):
        self.model.eval()
        y_preds = []
        y_truths = []
        for iter, (x, y) in enumerate(self.dataloader[name+'_loader'].get_iterator()):
            x, y = self._prepare_data(x, y)

            with torch.no_grad():
                preds = self.model(x,y)
            y_preds.append(preds)
            y = y[..., self.params['out_level']].squeeze()
            y_truths.append(y)

        y_preds = torch.cat(y_preds, axis=1)
        y_truths = torch.cat(y_truths, axis=1)
        y_preds = self.scaler.inverse_transform(y_preds,self.params['out_level'])
        y_truths = self.scaler.inverse_transform(y_truths,self.params['out_level'])
        mae, mape, rmse = metrics.metric(y_preds, y_truths)
        return mae, mape, rmse

    def ev_test(self, name):
        self.model.eval()
        y_preds = []
        y_truths = []
        for iter, (x, y) in enumerate(self.dataloader[name+'_loader'].get_iterator()):
            x, y = self._prepare_data(x, y)

            with torch.no_grad():
                preds = self.model(x,y)
            y_preds.append(preds)
            y = y[..., self.params['out_level']].squeeze()
            y_truths.append(y)

        y_preds = torch.cat(y_preds, axis=1)
        y_truths = torch.cat(y_truths, axis=1)

        mae = []
        mape = []
        rmse = []
        for i in range(self.seq_out_len):
            pred = self.scaler.inverse_transform(y_preds[i,...],self.params['out_level'])
            real = self.scaler.inverse_transform(y_truths[i,...],self.params['out_level'])
            results = metrics.metric(pred, real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, results[0], results[1], results[2]))
            mae.append(results[0])
            mape.append(results[1])
            rmse.append(results[2])
        return mae, mape, rmse


    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(self.device), y.to(self.device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x[-self.seq_in_len:,:,:,:self.in_dim]
        x = x.view(self.seq_in_len, batch_size, self.num_nodes * self.in_dim)

        y = y[-self.seq_out_len:,:,:,:self.in_dim]
        y = y.view(self.seq_out_len, batch_size, self.num_nodes, self.in_dim)

        # y = y[..., :1].view(-1, batch_size,
        #                                   self.num_nodes)
        # return x, y[:self.seq_out_len,...]
        return x, y
