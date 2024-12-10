import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import MSELoss


class MLPAutoencoder(nn.Module):
	def __init__(
			self,
			layer_sizes=(1, 2, 2, 4),
			enforce_inverse=True,
			init="kai",
			abs_inv_obj=True,
			jit=True,
			activation=nn.Mish(inplace=True),
			batchnorm=False,
	):
		super().__init__()
		self.layer_sizes = layer_sizes
		self.activation = activation
		self.abs_inv_obj = abs_inv_obj
		self.init = init
		layers_forward = []
		layers_reverse = []
		for i in range(len(self.layer_sizes) - 1):
			layers_forward.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
			layers_reverse.append(nn.Linear(layer_sizes[len(self.layer_sizes)-1-i], layer_sizes[len(self.layer_sizes)-2-i]))
			if i < len(layer_sizes) - 2:
				if batchnorm:
					layers_forward.append(nn.BatchNorm1d(layer_sizes[i + 1]))
					layers_reverse.append(nn.BatchNorm1d(layer_sizes[len(self.layer_sizes)-2-i]))
				layers_forward.append(self.activation)
				layers_reverse.append(self.activation)
		self.net_for = nn.Sequential(*layers_forward)
		self.net_rev = nn.Sequential(*layers_reverse)
		if jit:
			self.net_for = torch.jit.script(self.net_for)
			self.net_rev = torch.jit.script(self.net_rev)
		self.mse = MSELoss()
		self.inverse_loss = 0
		self.init_weights(forward=True)
		self.init_weights(forward=False)
		if enforce_inverse:
			self.register_forward_hook(self.inverse_objective)

	def inverse_objective(self, module, grad_input, grad_output):
		x = self.input_forward
		xhat = self.net_rev(self.net_for(x))
		if not self.abs_inv_obj:
			self.inverse_loss = self.mse(x, xhat)
		else:
			self.inverse_loss = self.mse(torch.abs(x), torch.abs(xhat))
		self.inverse_loss.backward()

	def forward(self, input: Tensor) -> Tensor:
		shape = list(input.shape)
		shape[-1] = self.layer_sizes[-1]
		x = input.reshape(-1, self.layer_sizes[0])
		y = self.net_for(x)
		self.input_forward = x
		return y.reshape(shape)

	def inverse(self, input: Tensor) -> Tensor:
		shape = list(input.shape)
		shape[-1] = self.layer_sizes[0]
		x = input.reshape(-1, self.layer_sizes[-1])
		y = self.net_rev(x)
		return y.reshape(shape)

	def init_weights(self, forward=True):
		if forward:
			net = self.net_for
		else:
			net = self.net_rev
		if not self.init:
			return
		for module in net.children():
			if hasattr(module, 'weight') and isinstance(module, torch.nn.Linear):
				# torch.nn.init.normal_(module.weight, mean=0.02, std=0.1)
				torch.nn.init.kaiming_normal_(module.weight)
	
	def init_weights_modified(self, forward=True):
		if forward:
			net = self.net_for
		else:
			net = self.net_rev
		if not self.init:
			return
		for module in net.children():
			if hasattr(module, 'weight'):
				# torch.nn.init.normal_(module.weight, mean=0.02, std=0.1)
				torch.nn.init.kaiming_normal_(module.weight)
	
	def reset_parameters(self):
		for name, param in self.named_parameters():
			if 'weight' in name:
				param = nn.init.kaiming_normal_(param.detach())
			elif 'bias' in name:
				param = nn.init.constant_(param.detach(), 0)


class BatchNorm(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.bn = nn.BatchNorm1d(*args, **kwargs)

	def forward(self, x):
		shape = x.shape
		x_r = x.reshape(np.prod(shape[:-1]), shape[-1])
		y_r = self.bn(x_r)
		y = y_r.reshape(shape)
		return y