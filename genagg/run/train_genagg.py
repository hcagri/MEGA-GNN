from torch_geometric.nn import GraphConv
from Runner import run
from genagg import GenAgg


def main():

	configs = {}

	configs["genagg"] = {
		"layer": GraphConv,
		"num_layers": 4,
		"hidden_size": 64,
		"agg_kwargs": {"layer_sizes": [1, 2, 2, 4], "init": "kai", "batchnorm": True},
		"layer_kwargs": {"aggr": GenAgg},
	}

	run(configs=configs, datasets=["PATTERN", "CLUSTER", "MNIST", "CIFAR10"],
		num_seeds=10, jobs_per_gpu=1, use_gpu=False, batch_size=32, single_thread=True,
		wandb_project=None, save=False)


if __name__ == '__main__':
	main()