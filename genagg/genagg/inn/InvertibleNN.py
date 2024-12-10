import torch
from torch import nn
from genagg.inn.Activation import *


class Layer(nn.Module):
	def __init__(self, dim, act):
		super().__init__()
		self.act = act
		self.A = nn.Parameter(torch.randn(dim,dim) * 0.3 + 1)
		self.Ainv = torch.inverse(self.A)
		self.b = nn.Parameter(torch.randn(dim, 1) * 1.0)
		self.A.register_hook(self.update_inv)

	def forward(self, x):
		return self.act(self.A @ x + self.b)

	def inverse(self, x):
		return self.Ainv @ (self.act.inverse(x)-self.b)

	def update_inv(self, *args):
		self.Ainv = torch.inverse(self.A)


class InvertibleNN(nn.Module):
	def __init__(self, dim=1):
		super().__init__()
		self.dim = dim
		# self.layers = nn.ModuleList([Layer(dim, act=SoftLin(fwd=True)), Layer(dim, act=SoftLin(fwd=True))])
		self.layers = nn.ModuleList([
			Layer(dim, act=SoftLin(fwd=True)),
			Layer(dim, act=SoftLin(fwd=False)),
			Layer(dim, act=Symlog(fwd=True)),
			Layer(dim, act=SoftLin(fwd=True)),
			Layer(dim, act=Symlog(fwd=False)),
			Layer(dim, act=SoftLin(fwd=False)),
			Layer(dim, act=SoftLin(fwd=False)),
			Layer(dim, act=Identity(fwd=True)),
		])

	def forward(self, x):
		x_shape = x.shape
		x = x.view(-1, self.dim).permute(1,0)
		for i in range(len(self.layers)):
			x = self.layers[i](x)
		return x.permute(1,0).view(*x_shape)

	def inverse(self, x):
		x_shape = x.shape
		x = x.view(-1, self.dim).permute(1,0)
		for i in reversed(range(len(self.layers))):
			x = self.layers[i].inverse(x)
		return x.permute(1,0).view(*x_shape)

	def dist_op(self, a, b, type=1):
		if type == 1:
			return self.inverse(self.forward(a) * self.forward(b))
		elif type == 0:
			return self.inverse(self.forward(a) + self.forward(b))
