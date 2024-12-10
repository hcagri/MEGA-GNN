from genagg import GenAgg
from genagg import InvertibleNN
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MeanAggregation
import torch

# torch.set_printoptions(sci_mode=False)

def test_sparse():
	import torch
	from genagg import GenAgg
	in_size = 8 # number of input nodes
	out_size = 4 # number of output nodes
	dim = 2 # dimension of each node
	x = torch.randn(in_size, dim) # input
	index = torch.randint(low=0, high=out_size, size=(in_size,)) # the output node for each input node to aggregate into
	class Identity:
		def forward(self, x):
			return x
		def inverse(self, x):
			return x
	agg = GenAgg(f=Identity(), a=torch.tensor(0.)) # initialise the aggregator
	y = agg(x=x, index=index) # compute the output
	print("x:", x)
	print("index:", index)
	print("y:", y.detach())


def test_dense():
	x = torch.rand(4,2)
	agg = GenAgg()
	y = agg(x, dim=-1)
	print("x:", x)
	print("y:", y.detach())


def test_inn():
	f = InvertibleNN()
	x = torch.linspace(-2,2,5)
	fx = f(x)
	xrec = f.inverse(fx)
	print("x:", x)
	print("f(x):", fx.detach())
	print("max error:", torch.max(torch.abs(x-xrec)).detach())


def test_gnn():
	edge_index = torch.tensor([[0, 1, 1, 2],
	                           [1, 0, 2, 1]], dtype=torch.long) # 2 x num_edges
	x = torch.rand(3,2) # num_nodes x feature_dim
	data = Data(x=x, edge_index=edge_index)
	agg = GenAgg() # init GenAgg
	gnn = GraphConv(in_channels=2, out_channels=1, aggr=agg) # init GNN with GenAgg
	y = gnn(x=x, edge_index=edge_index) # evaluate
	print("x:", x)
	print("y:", y.detach())


def test_dist():
	f = InvertibleNN()
	agg = GenAgg(f=f)
	x = torch.randn(10)
	c = torch.randn(1)
	yc1 = agg.dist_op(c, agg(x, dim=0))
	yc2 = agg(agg.dist_op(c, x), dim=0)
	print("c˚agg(xi):", yc1.detach())
	print("agg(c˚xi):", yc2.detach())


if __name__ == '__main__':
	test_dist()