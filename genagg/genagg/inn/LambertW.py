from scipy.special import lambertw as lambert_w_func
import torch

class LambertW(torch.autograd.Function):

	@staticmethod
	def forward(x):
		out = lambert_w_func(x.detach(), k=0)
		out = torch.as_tensor(out).real
		return out

	@staticmethod
	def setup_context(ctx, inputs, outputs):
		x, = inputs
		w = outputs
		ctx.save_for_backward(x, w)

	@staticmethod
	def backward(ctx, grad_output):
		x, w = ctx.saved_tensors
		grad = w / (x * (w + 1))
		return grad_output * grad

def lambertw(x):
	return LambertW.forward(x)

def run_lambert():
	eps = 1e-5
	x = torch.linspace(0.5, 2, 4, requires_grad=True)
	y = lambertw(x)
	out = y.sum()
	deriv = torch.autograd.grad(out, x)[0]
	num_deriv = (lambertw(x+eps) - y) / eps
	print(deriv)
	print(num_deriv)

if __name__ == "__main__":
	run_lambert()