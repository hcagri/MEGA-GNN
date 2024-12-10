import torch
from torch import nn
import inspect
from torch_geometric.nn.aggr import Aggregation
from genagg.MLPAutoencoder import MLPAutoencoder


class GenAgg(Aggregation):
	def __init__(
		self,
		f=None,
		a=None,
		b=None,
		**kwargs,
	):
		super().__init__()
		if f is None:
			self.f = MLPAutoencoder(layer_sizes=(1,2,2,4))
		elif inspect.isclass(f):
			self.f = f(**kwargs)
		else:
			self.f = f

		if a is None:
			self.a = nn.Parameter(torch.tensor(0.0))
		else:
			self.a = a
		if b is None:
			self.b = nn.Parameter(torch.tensor(0.0))
		else:
			self.b = b	
	

	def reset_parameters(self):
		# Reinitialize a and b
		nn.init.zeros_(self.a)
		nn.init.zeros_(self.b)
		
		# Reinitialize the parameters of self.f if it has reset_parameters method
		if hasattr(self.f, 'reset_parameters'):
			self.f.reset_parameters()
			

	def get_n(
		self,
		x,
		index=None,
		ptr=None,
		dim_size=None,
		dim=-2,
	):
		n = self.reduce(torch.ones_like(x), index, ptr, dim_size, dim, reduce='sum')
		n[n==0] = 1
		return n


	def forward(
		self,
		x,
		index=None,
		ptr=None,
		dim_size=None,
		dim=-2,
	):
		if isinstance(self.b, nn.Parameter) or self.b != 0:
			x_mean = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
			if index is None:
				x = x - self.b * x_mean.unsqueeze(dim)
			else:
				x = x - self.b * torch.index_select(input=x_mean, dim=dim, index=index)

		n = self.get_n(x=x, index=index, ptr=ptr, dim_size=dim_size, dim=dim)
		x = x.unsqueeze(-1)
		n = n.unsqueeze(-1)
		if dim < 0:
			dim -= 1

		y1 = self.f.forward(x)
		y2 = self.reduce(y1, index, ptr, dim_size, dim, reduce='mean')
		y3 = y2 * (n**self.a)
		z = self.f.inverse(y3)
		z = z.squeeze(-1)

		return z


	def dist_op(
		self, 
		a,
		b,
		type=1,
	):
		if type == 1:
			return self.f.inverse(self.f.forward(a) * self.f.forward(b))
		elif type == 0:
			return self.f.inverse(self.f.forward(a) + self.f.forward(b))



