from __future__ import division
import argparse
from dataset.data import *
from utils.metrics import *
from utils.process import *
import os
from trainer.ctrainer import CTrainer
from trainer.rtrainer import RTrainer
from nets.traverse_net import TraverseNet, TraverseNetst
from nets.stgcn_net import STGCNnet
from nets.graphwavenet import gwnet
from nets.astgcn_net import ASTGCNnet
from nets.dcrnn_net import DCRNNModel
import pickle
import dgl
import json
import random
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
torch.set_num_threads(3)

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def run(dataloader, device,params,net_params, adj_mx=None):
    scaler = dataloader['scaler']
    if net_params['model']=='traversenet':
        file = open(params['graph_path'], "rb")
        graph = pickle.load(file)
        relkeys = graph.keys()
        print([t[1] for t in graph.keys()])
        graph = dgl.heterograph(graph)
        graph = graph.to(device)
        # file = open('./data/randg/metr_ed1.pkl', "rb")
        # #file = open('./data/metr-Gstd.pkl', "rb")
        # ds = pickle.load(file)
        # for t in ds.keys():
        #     graph.edges[t].data['weight'] = ds[t]
        model = TraverseNet(net_params, graph, relkeys)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        num_training_steps = dataloader['train_loader'].num_batch*params['epochs']
        num_warmup_steps = int(num_training_steps*0.1)
        print('num_training_step:', num_training_steps)
        #lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        trainer = CTrainer(model, optimizer, masked_mae, dataloader, params, net_params['seq_out_len'], scaler, device)

    elif net_params['model']=='traversenet-ab':
        #traversenet-ab is a model setting in ablation study that interleaves temporal graphs with spatial graphs.
        file = open(params['graph_path'], "rb")
        graph = pickle.load(file)
        file.close()
        relkeys = graph.keys()
        print([t[1] for t in graph.keys()])
        graph = dgl.heterograph(graph)
        graph = graph.to(device)

        file1 = open(params['graph_path1'], "rb")
        graph1 = pickle.load(file1)
        file1.close()
        relkeys1 = graph1.keys()
        print([t[1] for t in graph1.keys()])
        graph1 = dgl.heterograph(graph1)
        graph1 = graph1.to(device)

        model = TraverseNetst(net_params, graph, graph1, relkeys, relkeys1)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        trainer = CTrainer(model, optimizer, masked_mae, dataloader, params, net_params['seq_out_len'], scaler, device)

    elif net_params['model']=='stgcn':
        adj_mx = sym_adj((adj_mx+adj_mx.transpose())/2)
        adj_mx = torch.Tensor(adj_mx.todense()).to(device)

        model = STGCNnet(net_params, adj_mx)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        trainer = CTrainer(model, optimizer, masked_mae, dataloader, params, net_params['seq_out_len'], scaler, device)

    elif net_params['model']=='graphwavenet':
        supports = [torch.Tensor(asym_adj(adj_mx).todense()).to(device),torch.Tensor(asym_adj(np.transpose(adj_mx)).todense()).to(device)]
        model = gwnet(net_params, device, supports)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        trainer = CTrainer(model, optimizer, masked_mae, dataloader, params, net_params['seq_out_len'], scaler, device)


    elif net_params['model']=='astgcn':
        L_tilde = scaled_Laplacian(adj_mx)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in
                            cheb_polynomial(L_tilde, net_params['K'])]
        model = ASTGCNnet(cheb_polynomials, net_params, device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        trainer = CTrainer(model, optimizer, masked_mae, dataloader, params, net_params['seq_out_len'], scaler, device)

    elif net_params['model']=='dcrnn':
        model = DCRNNModel(adj_mx, device, net_params)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], eps=params['epsilon'])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.1)
        trainer = RTrainer(model, optimizer, lr_scheduler, masked_mae, dataloader, params, net_params, scaler, device)

    elif net_params['model']=='gru':
        #the GRU model is equivalent to a DCRNN model with identity graph adjacency matrix.
        adj_mx = np.eye(net_params['num_nodes'])
        model = DCRNNModel(adj_mx, device, net_params)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], eps=params['epsilon'])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.1)
        trainer = RTrainer(model, optimizer, lr_scheduler, masked_mae, dataloader, params, net_params, scaler, device)

    else:
        print("model is not defined.")
        exit

    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)
    # nParams = sum([p.nelement() for p in model.start_conv.parameters()])
    # print('Number of model parameters for start conv is ', nParams)
    # nParams = sum([p.nelement() for p in model.transformer.parameters()])
    # print('Number of model parameters for transformer is ', nParams)
    print("start training...",flush=True)
    his_loss, train_time, val_time = [], [], []

    minl = 1e5

    for i in range(params['epochs']):
        train_loss,train_mape,train_rmse, traint = trainer.train_epoch()
        train_time.append(traint)

        valid_loss, valid_mape, valid_rmse, valt = trainer.val_epoch()
        val_time.append(valt)

        his_loss.append(valid_loss)
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch, Valid Time: {:.4f}/epoch'
        print(log.format(i, train_loss, train_mape, train_rmse, valid_loss, valid_mape, valid_rmse, traint, valt),flush=True)

        out_dir = params['out_dir']
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(model.state_dict(), '{}.pkl'.format(out_dir + "/epoch_" + str(i)))

        if valid_loss<minl:
            torch.save(trainer.model.state_dict(), '{}.pkl'.format(out_dir + "/epoch_best"))
            minl = valid_loss


    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    bestid = np.argmin(his_loss)
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    trainer.model.load_state_dict(torch.load('{}.pkl'.format(out_dir + "/epoch_best")))

    trmae, trmape, trrmse = trainer.ev_valid('train')
    vmae, vmape, vrmse = trainer.ev_valid('val')
    # tmae, tmape, trmse = trainer.ev_test('test')
    tmae, tmape, trmse = trainer.ev_valid('test')
    print('test', tmae,tmape,trmse)

    return trmae, trmape, trrmse, vmae, vmape, vrmse, tmae, tmape, trmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/traversenet.json')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--runs', type=int, help='number of runs')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--gpuid', type=int, help='device')
    parser.add_argument('--dataset', type=str, help='dataset')
    parser.add_argument('--graph_path', type=str, help='preprocessed graph for graph traversenet')
    parser.add_argument('--graph_path1', type=str, help='another preprocessed graph for a variant of graph traversenet in ablation study')
    parser.add_argument('--in_dim', type=int, help='dimension of inputs')
    parser.add_argument('--out_dim', type=int, help='dimension of outputs')

    parser.add_argument('--dim', type=int, help='hidden feature dimsion')
    parser.add_argument('--dropout', type=float, help='dropout rate')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay rate')

    parser.add_argument('--heads', type=int, help='number of heads for the attention mechanism')

    parser.add_argument('--num_layers', type=int, help='number of layers')
    parser.add_argument('--num_nodes', type=int, help='number of nodes')
    parser.add_argument('--num_rel', type=int, help='number of relation types in the preprocessed heterogenous graph')

    parser.add_argument('--seq_in_len', type=int, help='input sequence length')
    parser.add_argument('--seq_out_len', type=int, help='output sequence length')

    parser.add_argument('--out_dir', type=str, help='model save path')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--out_level', type=int, default=0, help='output level, 0 for traffic flow prediction, 2 for traffic speed prediction.(only applicable to PEMS-04 and PEMS-08)')


    args = parser.parse_args()
    print(args)

    with open(args.config) as f:
        config = json.load(f)

    if args.gpuid is not None:
        config['gpu']['id'] = args.gpuid

    params = config['params']
    if args.seed is not None:
        params['seed'] = args.seed
    if args.epochs is not None:
        params['epochs'] = args.epochs
    if args.runs is not None:
        params['runs'] = args.runs
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.dataset is not None:
        params['dataset'] = args.dataset
    if args.graph_path is not None:
        params['graph_path'] = args.graph_path
    if args.graph_path1 is not None:
        params['graph_path1'] = args.graph_path1
    if args.out_dir is not None:
        params['out_dir'] = args.out_dir
    if args.out_level is not None:
        params['out_level'] = args.out_level
    if args.lr is not None:
        params['lr'] = args.lr
    if args.weight_decay is not None:
        params['weight_decay'] = args.weight_decay

    net_params = config['net_params']
    if args.model is not None:
        net_params['model'] = args.model
    if args.dim is not None:
        net_params['dim'] = args.dim
    if args.in_dim is not None:
        net_params['in_dim'] = args.in_dim
    if args.out_dim is not None:
        net_params['out_dim'] = args.out_dim
    if args.num_layers is not None:
        net_params['num_layers'] = args.num_layers
    if args.num_nodes is not None:
        net_params['num_nodes'] = args.num_nodes
    if args.seq_in_len is not None:
        net_params['seq_in_len'] = args.seq_in_len
    if args.seq_out_len is not None:
        net_params['seq_out_len'] = args.seq_out_len
    if args.num_rel is not None:
        net_params['num_rel'] = args.num_rel

    if args.dropout is not None:
        net_params['dropout'] = args.dropout
    if args.heads is not None:
        net_params['heads'] = args.heads



    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    dataloader = load_data(params['batch_size'], "data/"+params['dataset']+".pkl", device)
    adj_mx = np.array(dataloader['adj']+torch.eye(net_params['num_nodes']))

    trmae, trmape, trrmse, vmae, vmape, vrmse, mae, mape, rmse = [], [], [], [], [], [], [], [], []
    for i in range(params['runs']):
        tm1, tm2, tm3, vm1, vm2, vm3, m1, m2, m3 = run(dataloader, device,params,net_params, adj_mx)
        trmae.append(tm1)
        trmape.append(tm2)
        trrmse.append(tm3)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)


    print('\n\nResults for {:d} runs\n\n'.format(params['runs']))
    #train data
    print('train\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(trmae),np.mean(trrmse),np.mean(trmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(trmae),np.std(trrmse),np.std(trmape)))
    print('\n\n')

    #valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae),np.std(vrmse),np.std(vmape)))
    print('\n\n')

    #test data
    print('test\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(mae),np.mean(rmse),np.mean(mape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(mae),np.std(rmse),np.std(mape)))
    print('\n\n')



if __name__ == "__main__":
    main()






