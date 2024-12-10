import torch
from torch.optim import Adam
from torch.nn import MSELoss
from Visualiser import Visualiser
from genagg import InvertibleNN

g = lambda x: torch.exp(x)

def run():
	vis = Visualiser()
	f = InvertibleNN(dim=1)
	optim = Adam(f.parameters(), lr=1e-3)
	lossfn = MSELoss()
	T = 100000
	batch = 64
	for t in range(T):
		x = torch.randn(batch,1)
		y = f(x)
		y_true = g(x)
		loss = lossfn(y, y_true)
		loss.backward()
		optim.step()
		optim.zero_grad()
		if t % 10 == 0:
			vis.clear()
			vis.update(f.forward, lim=[-5, 5, -5, 5], step=0.01, visdim=[0], ninputs=1)
			vis.update(f.inverse, lim=[-5, 5, -5, 5], step=0.01, visdim=[0], ninputs=1, color="red")
			vis.update(g, lim=[-5, 5, -5, 5], step=0.01, visdim=[0], ninputs=1, linestyle="dashed")

if __name__ == "__main__":
	run()