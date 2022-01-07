from dataset.data import *

batch_size = 64
num_nodes = 170
path='./data/PEMS08/PEMS08.npz'
adjpath='./data/PEMS08/PEMS08.csv'
idpath='./data/PEMS08/PEMS08.txt'
op = PemsData(num_nodes, path, adjpath, None)
op.prcoess('./data/PEMS-D8.pkl')
_, srclist, tgtlist, distlist = op.load_graph()
g, _ = process_t_graph(srclist, tgtlist, distlist, 12, num_nodes, window=12)
file = open('./data/PEMS-D8-Gt.pkl', "wb")
pickle.dump(g, file)
#
#
batch_size = 64
num_nodes = 358
path='./data/PEMS03/PEMS03.npz'
adjpath='./data/PEMS03/PEMS03.csv'
idpath='./data/PEMS03/PEMS03.txt'
op = PemsData(num_nodes, path, adjpath, idpath)
op.prcoess('./data/PEMS-D3.pkl')
_, srclist, tgtlist, distlist = op.load_graph()
g, _ = process_t_graph(srclist, tgtlist, distlist, 12, num_nodes, window=12)
file = open('./data/PEMS-D3-Gt.pkl', "wb")
pickle.dump(g, file)
#
batch_size = 64
num_nodes = 307
path='./data/PEMS04/PEMS04.npz'
adjpath='./data/PEMS04/PEMS04.csv'
idpath='./data/PEMS04/PEMS04.txt'
op = PemsData(num_nodes, path, adjpath, None)
op.prcoess('./data/PEMS-D4.pkl')
_, srclist, tgtlist, distlist = op.load_graph()
g, _ = process_t_graph(srclist, tgtlist, distlist, 12, num_nodes, window=12)
file = open('./data/PEMS-D4-Gt.pkl', "wb")
pickle.dump(g, file)
#
