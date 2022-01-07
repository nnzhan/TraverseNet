from dataset.data import *

num_nodes = 20
op = PoxData("./dataset1/chickenpox.json",'X')
op.prcoess('./pox.pkl')
_, srclist, tgtlist, distlist = op.load_graph1()
g, _ = process_t_graph(srclist, tgtlist, distlist, 12, num_nodes, window=12)
file = open('./pox-Gt.pkl', "wb")
pickle.dump(g, file)

num_nodes = 319
op = WindmillData("./dataset1/windmill_output.json")
op.prcoess('./windmill.pkl')
_, srclist, tgtlist, distlist = op.load_graph()
g, _ = process_t_graph(srclist, tgtlist, distlist, 12, num_nodes, window=12)
file = open('./windmill-Gt.pkl', "wb")
pickle.dump(g, file)