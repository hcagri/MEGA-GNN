import numpy as np
import wandb
import inspect
from typing import Optional
from torch_geometric.typing import Adj
from torch import Tensor
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from torch_geometric.nn.aggr import *
import gym
from gym import spaces
from matplotlib import pyplot as plt

from genagg import GenAgg
from genagg import MLPAutoencoder

project = None
user = ""

def experiments():
	trials = {
		"genagg": [
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "max", "mixfunc": lambda local, neighbours: np.max(neighbours, axis=0)},
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "min", "mixfunc": lambda local, neighbours: np.min(neighbours, axis=0)},
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "max-mag", "mixfunc": lambda local, neighbours: np.max(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "min-mag", "mixfunc": lambda local, neighbours: np.min(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "mean", "mixfunc": lambda local, neighbours: np.mean(neighbours, axis=0)},
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "sum", "mixfunc": lambda local, neighbours: np.sum(neighbours, axis=0)},
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "std", "mixfunc": lambda local, neighbours: np.std(neighbours, axis=0)},
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "2-norm", "mixfunc": lambda local, neighbours: np.linalg.norm(neighbours, axis=0)},
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "rms", "mixfunc": lambda local, neighbours: np.sqrt(np.mean(neighbours ** 2, axis=0))},
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "prod-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "geom-mean-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0) ** (1 / neighbours.shape[0])},
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "harm-mean-abs", "mixfunc": lambda local, neighbours: neighbours.shape[0] / np.sum(1 / np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": GenAgg, "f": MLPAutoencoder, "layer_sizes": (1,2,4,2,1)}, "agg": "log-sum-exp", "mixfunc": lambda local, neighbours: np.log(np.sum(np.exp(neighbours), axis=0))},
		],
		"mean": [
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "max", "mixfunc": lambda local, neighbours: np.max(neighbours, axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "min", "mixfunc": lambda local, neighbours: np.min(neighbours, axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "max-mag", "mixfunc": lambda local, neighbours: np.max(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "min-mag", "mixfunc": lambda local, neighbours: np.min(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "mean", "mixfunc": lambda local, neighbours: np.mean(neighbours, axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "sum", "mixfunc": lambda local, neighbours: np.sum(neighbours, axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "std", "mixfunc": lambda local, neighbours: np.std(neighbours, axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "2-norm", "mixfunc": lambda local, neighbours: np.linalg.norm(neighbours, axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "rms", "mixfunc": lambda local, neighbours: np.sqrt(np.mean(neighbours ** 2, axis=0))},
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "prod-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "geom-mean-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0) ** (1 / neighbours.shape[0])},
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "harm-mean-abs", "mixfunc": lambda local, neighbours: neighbours.shape[0] / np.sum(1 / np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation()}, "agg": "log-sum-exp", "mixfunc": lambda local, neighbours: np.log(np.sum(np.exp(neighbours), axis=0))},
		],
		"poweragg": [
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "max", "mixfunc": lambda local, neighbours: np.max(neighbours, axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "min", "mixfunc": lambda local, neighbours: np.min(neighbours, axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "max-mag", "mixfunc": lambda local, neighbours: np.max(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "min-mag", "mixfunc": lambda local, neighbours: np.min(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "mean", "mixfunc": lambda local, neighbours: np.mean(neighbours, axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "sum", "mixfunc": lambda local, neighbours: np.sum(neighbours, axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "std", "mixfunc": lambda local, neighbours: np.std(neighbours, axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "2-norm", "mixfunc": lambda local, neighbours: np.linalg.norm(neighbours, axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "rms", "mixfunc": lambda local, neighbours: np.sqrt(np.mean(neighbours ** 2, axis=0))},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "prod-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "geom-mean-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0) ** (1 / neighbours.shape[0])},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "harm-mean-abs", "mixfunc": lambda local, neighbours: neighbours.shape[0] / np.sum(1 / np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom(learn=True)}, "agg": "log-sum-exp", "mixfunc": lambda local, neighbours: np.log(np.sum(np.exp(neighbours), axis=0))},
		],
		"softmaxagg": [
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "max", "mixfunc": lambda local, neighbours: np.max(neighbours, axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "min", "mixfunc": lambda local, neighbours: np.min(neighbours, axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "max-mag", "mixfunc": lambda local, neighbours: np.max(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "min-mag", "mixfunc": lambda local, neighbours: np.min(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "mean", "mixfunc": lambda local, neighbours: np.mean(neighbours, axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "sum", "mixfunc": lambda local, neighbours: np.sum(neighbours, axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "std", "mixfunc": lambda local, neighbours: np.std(neighbours, axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "2-norm", "mixfunc": lambda local, neighbours: np.linalg.norm(neighbours, axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "rms", "mixfunc": lambda local, neighbours: np.sqrt(np.mean(neighbours ** 2, axis=0))},
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "prod-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "geom-mean-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0) ** (1 / neighbours.shape[0])},
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "harm-mean-abs", "mixfunc": lambda local, neighbours: neighbours.shape[0] / np.sum(1 / np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation(learn=True)}, "agg": "log-sum-exp", "mixfunc": lambda local, neighbours: np.log(np.sum(np.exp(neighbours), axis=0))},
		],
	}
	default = {
		"model": AggrGNN,
		"model_kwargs": {"aggr": GenAgg},
		"mixfunc": None,
		"agg": "n/a",
		"log_f": True,
		"train_iter": 10000,
		"epoch_size": 1024,
		"batch_size": 1024,
		"n_agents": 8,
		"obs_size": 6,
		"runs": 10,
	}
	n_runs = default["runs"]
	for r in range(n_runs):
		for name, trial in trials.items():
			if not isinstance(trial, list):
				trial = [trial]
			for cfg in trial:
				config = default.copy()
				config.update(cfg)
				config["name"] = name
				if r >= config.get("runs", n_runs):
					break
				del config["runs"]
				run(**config)


def run(
	mixfunc = lambda local, neighbours: np.min(neighbours, axis=0),
	name = "n/a",
	agg = "n/a",
	model = None,
	model_kwargs = {},
	train_iter = 10000,
	epoch_size = 1024,
	batch_size = 64,
	n_agents = 16,
	obs_size = 1,
	log_f = False,
	):

	if inspect.isclass(model):
		model = model(**model_kwargs)

	if torch.cuda.is_available():
		device = "cuda:0"
	else:
		device = "cpu"
	model = model.to(device)

	env_config = {
		"n_agents": n_agents,
		"obs_size": obs_size,
		"mixfunc": mixfunc,
	}

	env = PropEnv(**env_config)

	env_config.update({"agg": agg})

	if project is not None:
		wandb.init(
			entity=user,
			project=project,
			group=f"{name}",
			config=env_config,
		)

	if not isinstance(model_kwargs["aggr"], MeanAggregation):
		optimiser = torch.optim.Adam(model.parameters())
	loss_func = torch.nn.MSELoss()

	if log_f:
		model.aggr.visualize = True

	for t in range(train_iter):

		data = generate_data(env=env, steps=epoch_size)

		A = data["adj"][:, -1]
		x = data["obs"][:, -1]
		y = data["y"][:, -1]

		dataobj = adj2geom(A=A, x=x, y=y).to(device)
		loader = DataLoader(dataobj, batch_size=batch_size)

		avg_loss = 0
		avg_corr = 0

		for batch in loader:

			y_preds = model(x=batch.x, edge_index=batch.edge_index)

			loss = loss_func(y_preds, batch.y.float())
			if not isinstance(model_kwargs["aggr"], MeanAggregation):
				loss.backward()
				optimiser.step()
				optimiser.zero_grad()

			avg_loss += loss.item() * batch.num_graphs
			corr = np.corrcoef(
				y_preds.detach().cpu().numpy().reshape(-1), batch.y.detach().cpu().numpy().reshape(-1), rowvar=False
			)[0, 1]
			avg_corr += corr * batch.num_graphs

		alpha = 0
		beta = 0
		f_plot = None
		fig = None
		if isinstance(model.aggr, GenAgg):
			alpha = model.aggr.a.data.item()
			beta = model.aggr.b.data.item()
			if t % 10 == 0:
				x = torch.linspace(-2,2,200).view(-1,1)
				fx = model.aggr.f(x)
				fy = model.aggr.f.inverse(x)
				fig, ax = plt.subplots()
				ax.plot(x,fx.detach(),'b')
				ax.plot(x,fy.detach(),'r')
		elif isinstance(model.aggr, PowerMeanAggregationCustom):
			alpha = model.aggr.p.data.item()
		elif isinstance(model.aggr, SoftmaxAggregation):
			alpha = model.aggr.t.data.item()
		avg_loss /= len(loader.dataset)
		avg_corr /= len(loader.dataset)
		if project is not None:
			wandb.log({
				"loss": avg_loss,
				"corr": avg_corr,
				"alpha": alpha,
				"beta": beta,
				"f": fig,
			})
		if fig is not None:
			plt.close(fig)

	if project is not None:
		wandb.finish()


class AggrGNN(MessagePassing):
	def __init__(self, aggr, *args, **kwargs):
		super().__init__()
		if inspect.isclass(aggr):
			self.aggr = aggr(*args, **kwargs)
		else:
			self.aggr = aggr

	def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
		global_embed = self.propagate(edge_index=edge_index, x=x)
		return global_embed

	def aggregate(
		self,
		inputs: Tensor,
		index: Tensor,
		ptr: Optional[torch.Tensor] = None,
		dim_size: Optional[int] = None,
	) -> Tensor:
		y = self.aggr(x=inputs, index=index, dim_size=dim_size)
		return y

	def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
		return x_j


def generate_data(env, steps):
	observations = [None] * steps
	adjacencies = [None] * steps
	ys = [None] * steps
	i = 0
	t = 0
	while t < steps:
		done = False
		obs_i = []
		adj_i = []
		y_i = []
		while done is False:
			action = np.stack(env.action_space.sample())
			obs, reward, done, info = env.step(action)
			obs_i.append(obs["obs"])
			adj_i.append(obs["adj"])
			y_i.append(obs["prop"])
			t += 1
		observations[i] = np.stack(obs_i)
		adjacencies[i] = np.stack(adj_i)
		ys[i] = np.stack(y_i)
		i += 1
		env.reset()
	observations = np.stack(observations[:i])
	adjacencies = np.stack(adjacencies[:i])
	ys = np.stack(ys[:i])
	return {"obs": observations, "adj": adjacencies, "y": ys}


class PropEnv(gym.Env):
	def __init__(self, **kwargs):
		self.n_agents = 16
		self.n_actions = 0
		self.episode_length = 1
		self.density = 0.3
		self.mixfunc = lambda local, neighbours: np.max(neighbours, axis=0)[0]
		self.obs_size = 1
		# p = 0: np.prod(neighbours) ** (1/len(neighbours))
		# p = -1: len(neighbours) / np.sum(1/neighbours, axis=0)
		# p = 1: np.mean(neighbours, axis=0)
		# p = -inf: np.min(neighbours, axis=0)
		# p = inf: np.max(neighbours, axis=0)

		self.set_params(kwargs)

		self.observation_space_single = spaces.Box(
			low=np.array([-np.inf] * self.obs_size), high=np.array([np.inf] * self.obs_size)
		)
		self.action_space_single = spaces.Box(
			low=0.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
		)

		self.action_space = spaces.Tuple(
			[self.action_space_single for _ in range(self.n_agents)]
		)
		self.observation_space = spaces.Dict(
			{
				"obs": spaces.Tuple(
					[self.observation_space_single for _ in range(self.n_agents)]
				),
				"prop": spaces.Tuple(
					[self.observation_space_single for _ in range(self.n_agents)]
				),
				"adj": spaces.Tuple(
					[spaces.MultiBinary(self.n_agents) for _ in range(self.n_agents)]
				),
			}
		)

		PropEnv.reset(self)

	def set_params(self, params):
		for name, value in params.items():
			setattr(self, name, value)

	def mix_states(self, local_state):
		neighbour_states = self.adjacency[:, :, None] * local_state[None, :, :]
		new_state = np.array(
			[
				self.mixfunc(local_state[i], neighbour_states[i, self.adjacency[i], :])
				for i in range(local_state.shape[0])
			]
		)
		return new_state

	def step(self, action):
		self.t += 1
		state = self.state
		adj = self.adjacency
		prop = self.mix_states(state)
		self.state = prop
		return (
			{
				"obs": tuple(state.astype(np.float32)),
				"prop": tuple(prop.astype(np.float32)),
				"adj": tuple(adj.astype(np.byte)),
			},
			0,
			self.t >= self.episode_length,
			{},
		)

	def reset(self):
		self.t = 0
		self.state = np.random.randn(self.n_agents, self.obs_size)
		self.adjacency = self.gen_adjacency()
		obs = {
			"obs": tuple(self.state.astype(np.float32)),
			"prop": tuple(np.zeros((self.n_agents, 1), dtype=np.float32)),
			"adj": tuple(self.adjacency.astype(np.byte)),
		}
		return obs

	def gen_adjacency(self):
		orig_density = np.sqrt(self.density)
		while True:
			A = torch.rand(self.n_agents, self.n_agents) < orig_density
			torch.diagonal(A).fill_(0)
			A = A * A.T
			if torch.any(torch.sum(A, dim=0) == 0).item():
				continue
			return A.numpy()


def adj2geom(A, **kwargs):
    A = torch.as_tensor(A)
    for name, val in kwargs.items():
        kwargs[name] = torch.as_tensor(val)
    numdim = len(A.shape)
    numbatchdim = numdim - 2
    batchdimsize = int(np.prod(A.shape[:numbatchdim]))
    A = A.reshape(batchdimsize, *A.shape[numbatchdim:])
    for name, val in kwargs.items():
        kwargs[name] = val.reshape(batchdimsize, *val.shape[numbatchdim:])
    return dense_to_geometric(A=A, **kwargs)


def dense_to_geometric(A, **kwargs):
    edge_list = [from_scipy_sparse_matrix(coo_matrix(A[i])) for i in range(A.shape[0])]
    arg_slice = lambda i: {key: val[i] for key, val in kwargs.items()}
    data_list = [
        Data(edge_index=edge_index, edge_attr=edge_attr, **(arg_slice(i)))
        for i, (edge_index, edge_attr) in enumerate(edge_list)
    ]
    return Batch.from_data_list(data_list)


class PowerMeanAggregationCustom(Aggregation):

	def __init__(self, p: float = 1.0, learn: bool = False):
		super().__init__()
		self._init_p = p
		self.learn = learn
		self.p = nn.Parameter(torch.Tensor(1)) if learn else p
		self.reset_parameters()

	def reset_parameters(self):
		if isinstance(self.p, Tensor):
			self.p.data.fill_(self._init_p)

	def forward(self, x: Tensor, index: Optional[Tensor] = None,
				ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
				dim: int = -2) -> Tensor:
		x = x.clamp(min=1e-6, max=100).pow(self.p)
		out = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
		out = out.clamp(min=1e-6, max=100).pow(1. / self.p)
		return out


if __name__ == "__main__":
	experiments()
