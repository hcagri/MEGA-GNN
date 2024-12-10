import torch
from torch import nn
from genagg.inn.LambertW import lambertw

class Activation(nn.Module):
	def __init__(self, fwd=True):
		super().__init__()
		self.fwd = fwd

	def forward(self, x):
		if self.fwd:
			return self.f(x)
		else:
			return self.finv(x)

	def inverse(self, x):
		if self.fwd:
			return self.finv(x)
		else:
			return self.f(x)


class Symlog(Activation):
	def f(self, x):
		return torch.sign(x) * torch.log(torch.abs(x) + 1)

	def finv(self, x):
		return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class LeakyRelu(Activation):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.a = nn.Parameter(torch.tensor(0.1))

	def f(self, x):
		self.constraint()
		return torch.maximum(x, torch.tensor(0.)) + self.a * torch.minimum(x, torch.tensor(0.))

	def finv(self, x):
		return torch.maximum(x, torch.tensor(0.)) + (1 / self.a) * torch.minimum(x, torch.tensor(0.))

	def constraint(self, *args):
		self.a.data = torch.abs(self.a)


class SoftLin(Activation):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.a = nn.Parameter(torch.tensor(0.75))
		self.b = nn.Parameter(torch.tensor(0.25))

	def f(self, x):
		self.constraint()
		y = torch.zeros_like(x)
		mask = (x >= 0)
		y[mask] = (self.a + self.b) * x[mask]
		y[~mask] = torch.exp(self.a * x[~mask]) + self.b * x[~mask] - 1
		return y

	def finv(self, x):
		y = torch.zeros_like(x)
		mask = (x >= 0)
		y[mask] = 1 / (self.a + self.b) * x[mask]
		y[~mask] = -lambertw(self.a / self.b * torch.exp(self.a / self.b * (x[~mask] + 1))).float() / self.a + (x[~mask] / self.b) + (1 / self.b)
		return y

	def constraint(self):
		self.b.data = torch.maximum(self.b, torch.tensor(0.1))
		self.a.data = torch.abs(self.a)


class Identity(Activation):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def f(self, x):
		return x

	def finv(self, x):
		return x


if __name__ == "__main__":
	import matplotlib
	matplotlib.use('TkAgg')
	from matplotlib import pyplot as plt

	f = SoftLin()
	x = torch.linspace(-5, 0, 6)
	y = f(x)
	x_rec = f.inverse(y)

	# test = 1 + x - lambertw(torch.exp(1+x))
	print(x, x_rec, y)
	plt.show()
