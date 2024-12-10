from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import torch


class Visualiser:

	def __init__(self, ax=None):
		self.ax = ax
		plt.ion()


	def clear(self):
		plt.cla()


	def update(self, func, lim=[-1,1,-1,1], step=0.1, color="cyan", linestyle="solid", visdim=[0,1], ninputs=2, defaultval=0., label=None, cmap=cm.viridis):
		if self.ax is None:
			self.ax = plt.gca()
		x = torch.tensor(np.arange(lim[0],lim[1],step))
		x = x.float().view(-1, 1)
		z = func(x)
		x = x.detach().numpy()
		z = z.view(x.shape).detach().numpy()
		self.ax.plot(x, z, color="tab:"+color, linestyle=linestyle, label=label)
		self.ax.set_xlim(lim[0],lim[1])
		if len(lim) > 2:
			self.ax.set_ylim(lim[2], lim[3])

		plt.draw()
		plt.pause(0.01)


if __name__ == "__main__":
	vis = Visualiser()
	func = lambda x: x[:,0]**2 + x[:,1]
	vis.update(func)