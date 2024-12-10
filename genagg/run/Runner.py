import sys
import os
import pdb
import random
import itertools
import time
import pandas
from typing import Any, Dict, List
import inspect
import copy

import torch
import torch.multiprocessing as multiprocessing
import torch_geometric
from genagg import GenAgg
from torch_geometric.loader import DataLoader
import wandb
import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

model_path = "saved/{name}_{project}.pt"

user = ""


def get_loss_fn(task):
    if task == "graph_class":
        return torch.nn.functional.cross_entropy
    elif task == "node_class":
        return torch.nn.functional.cross_entropy
    elif task == "graph_regr":
        return torch.nn.functional.mse_loss

def class_acc(yhat, y):
    class_pred = torch.argmax(yhat, dim=-1)
    n_corr = torch.sum(class_pred == y)
    tot = y.numel()
    acc = n_corr / tot
    return acc

def get_accuracy_fn(task):
    if task == "graph_class":
        return class_acc
    elif task == "node_class":
        return class_acc
    elif task == "graph_regr":
        return lambda yhat, y: 0



def build_gnn(
    layer,
    input_size,
    output_size,
    task="node_class",
    num_layers=4,
    hidden_size=64,
    agg_kwargs={},
    layer_kwargs={},
):
    hidden_layers = []

    for i in range(num_layers):

        if isinstance(layer_kwargs, dict):
            layer_kwargs_dict = copy.deepcopy(layer_kwargs.copy())
        else:
            layer_kwargs_dict = layer_kwargs(i)

        if isinstance(agg_kwargs, dict):
            agg_kwargs_dict = copy.deepcopy(agg_kwargs)
        else:
            agg_kwargs_dict = agg_kwargs(i)

        if isinstance(hidden_size, int):
            in_dim = hidden_size
            out_dim = hidden_size
        elif isinstance(hidden_size, list):
            assert (len(hidden_size) == num_layers-1)
            layer_sizes = [input_size] + hidden_size + [output_size]
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]

        aggr = layer_kwargs_dict.get("aggr", "mean")
        if inspect.isclass(aggr):
            aggr = aggr(**agg_kwargs_dict)
        del layer_kwargs_dict["aggr"]
        layeri = layer(in_channels=in_dim, out_channels=out_dim, aggr=aggr, **layer_kwargs_dict)

        activation = layer_kwargs_dict.get("activation", torch.nn.Mish(inplace=True))

        hidden_layers.append((layeri, "x, e -> x"))
        hidden_layers.append((activation, "x -> x"))



    core = [
        (torch.nn.Linear(input_size, hidden_size), "x -> x"),
        *hidden_layers,
    ]
    # Output layers
    if task == "node_class":
        core += [
            (torch.nn.Linear(hidden_size, output_size), "x -> x"),
            (activation, "x -> x"),
            (torch.nn.Linear(output_size, output_size), "x -> x"),
            (activation, "x -> x"),
            (torch.nn.Linear(output_size, output_size), "x -> x"),
            (activation, "x -> x"),
            (torch.nn.Linear(output_size, output_size), "x -> x"),
        ]
    elif task == "graph_class":
        core += [
            (torch_geometric.nn.global_mean_pool, "x, b, size -> x"),
            (torch.nn.Linear(hidden_size, hidden_size), "x -> x"),
            (activation, "x -> x"),
            (torch.nn.Linear(hidden_size, hidden_size), "x -> x"),
            (activation, "x -> x"),
            (torch.nn.Linear(hidden_size, hidden_size), "x -> x"),
            (activation, "x -> x"),
            (torch.nn.Linear(hidden_size, output_size), "x -> x"),
        ]
    elif task == "graph_regr":
        core += [
            (torch_geometric.nn.global_mean_pool, "x, b, size -> x"),
            (torch.nn.Linear(hidden_size, hidden_size), "x -> x"),
            (activation, "x -> x"),
            (torch.nn.Linear(hidden_size, hidden_size), "x -> x"),
            (activation, "x -> x"),
            (torch.nn.Linear(hidden_size, hidden_size), "x -> x"),
            (activation, "x -> x"),
            (torch.nn.Linear(hidden_size, 1), "x -> x"),
        ]
    else:
        raise NotImplementedError()

    model = torch_geometric.nn.Sequential("x, e, b, size", core)

    model.name = f"{layer.__name__}_{num_layers}x{hidden_size}"

    hparams = {
        "hidden_size": hidden_size,
        "task": task,
        "hidden_layers": num_layers,
        "layer_type": layer.__name__[:63],
        "aggr": aggr,
    }
    return model, hparams


def run_exp(
    name: str,
    config: dict,
    dataset: str,
    device_q: multiprocessing.Queue,
    use_gpu: bool,
    seed: int,
    wandb_project: str,
    batch_size: int = 32,
    epochs: int = 1000,
    save: bool = False,
):
    os.environ["WANDB_START_METHOD"] = "thread"
    torch.manual_seed(seed)
    random.seed(seed)
    # Select device
    device_id = 0
    if use_gpu:
        if torch.cuda.is_available():
            device_id = device_q.get(timeout=1)
            device = f"cuda:{device_id}"
        else:
            device = "mps"
            use_gpu = False
    else:
        device = "cpu"

    # Load dataset
    dset_info = get_dataset(dataset)
    dset = dset_info["dataset"]
    dset_type = dset_info["type"]


    # Construct dataset-specific models
    config["task"] = dset_type
    config["input_size"] = max(dset.num_node_features, 1)
    config["output_size"] = dset.num_classes
    if config["layer"] == torch_geometric.nn.PNAConv:
        config["layer_kwargs"]["deg"] = compute_dataset_degrees(dset)

    gnn, hparams = build_gnn(**config)
    gnn = gnn.to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=0.001)
    loss_fn = get_loss_fn(dset_type)
    accuracy_fn = get_accuracy_fn(dset_type)
    print(f"{dataset}, {name} on {device}")

    # Dataloaders
    eval_data_size = 10 # 1/10th of the dataset
    if len(dset) == 1:
        from torch_geometric.transforms import RandomNodeSplit
        transform = RandomNodeSplit(num_val=0.1, num_test=0.)
        masked_data = transform(dset[0])
        loader = [torch_geometric.data.Batch.from_data_list([masked_data]).to(device)]
        eval_data = torch_geometric.data.Batch.from_data_list([masked_data]).to(device)
    else:
        mask = torch.zeros(len(dset)).bool()
        mask[::eval_data_size] = 1
        loader = DataLoader(dset[~mask], batch_size=batch_size, shuffle=True)
        eval_data = next(iter(DataLoader(dset[mask], batch_size=int(mask.sum())))).to(device)
    # Pretrial init
    if wandb_project is not None:
        wandb.init(
            entity=user,
            project=wandb_project,
            group=dataset,
            job_type=name,
            config={
                "name": name,
                "dataset": dataset,
                **hparams,
                **config["layer_kwargs"],
                "seed": seed,
            },
            reinit=True,
        )

    train_batches = 0
    train_samples = 0

    for epoch in tqdm.trange(epochs, position=device_id):
        for data in loader:
            data = data.to(device)

            if data.x is None:
                x = torch.ones(data.num_nodes, 1, device=device)
            else:
                if data.x.dtype != torch.float:
                    data.x = data.x.float()

                x = data.x
                if x.dim() == 1:
                    x = x.unsqueeze(1)

            out = gnn(x, data.edge_index, data.batch, data.num_graphs)

            y = data.y
            if hasattr(data, "train_mask"):
                out = out[data.train_mask]
                y = y[data.train_mask]

            loss = loss_fn(out, y)
            accuracy = accuracy_fn(out, y)
            pre_backward_time = time.time()

            loss.backward()

            backward_time = time.time() - pre_backward_time
            opt.step()
            opt.zero_grad()

            # Log p,a
            if wandb_project is not None:
                wandb.log(
                    {
                        "train_loss": loss,
                        "train_accuracy": accuracy,
                        "epoch": epoch,
                        "batch": train_batches,
                        "sample": train_samples,
                        "backward_time": backward_time,
                        "seed": seed,
                    },
                    commit=True,
                )

            train_batches += 1
            train_samples += data.num_graphs

        # Eval
        with torch.no_grad():
            if eval_data.x is None:
                x = torch.ones(eval_data.num_nodes, 1, device=device)
            else:
                x = eval_data.x
            out = gnn(
                x,
                eval_data.edge_index,
                eval_data.batch,
                eval_data.num_graphs,
            )

        y = eval_data.y
        if hasattr(eval_data, "val_mask"):
            out = out[data.val_mask]
            y = y[data.val_mask]

        test_loss = loss_fn(out, y)
        test_accuracy = accuracy_fn(out, y)

        if wandb_project is not None:
            wandb.log(
                {
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                },
                commit=False,
            )


    if save:
        dir_name = os.path.split(model_path)[0]
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        model_state_dict = gnn.state_dict()
        torch.save(model_state_dict, model_path.format(name=name.replace('/','-'), project=dataset))

    wandb.finish()
    device_q.put(device_id)
    time.sleep(5)


def get_dataset(name):
    if name.upper() == "CLUSTER":
        from torch_geometric.datasets import GNNBenchmarkDataset
        dataset = GNNBenchmarkDataset(root="/tmp/dataset", name="CLUSTER")
        problem = "node_class"
    elif name.upper() == "PATTERN":
        from torch_geometric.datasets import GNNBenchmarkDataset
        dataset = GNNBenchmarkDataset(root="/tmp/dataset", name="PATTERN")
        problem = "node_class"
    elif name.upper() == "CIFAR10":
        from torch_geometric.datasets import GNNBenchmarkDataset
        dataset = GNNBenchmarkDataset(root='/tmp/dataset', name='CIFAR10')
        problem = "graph_class"
    elif name.upper() == "MNIST":
        from torch_geometric.datasets import GNNBenchmarkDataset
        dataset = GNNBenchmarkDataset(root='/tmp/dataset', name='MNIST')
        problem = "graph_class"
    
    return {"dataset": dataset, "type": problem}



def compute_dataset_degrees(dataset):
    # Compute degs for PNAConv
    max_degree = -1
    for data in dataset:
        d = torch_geometric.utils.degree(
            data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
        )
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = torch_geometric.utils.degree(
            data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
        )
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


def run_exp_args(args):
    run_exp(*args)


def run(configs: Dict[str, Dict[str, Any]], 
        datasets: List[str],
        batch_size: int = 32,
        epochs: int = 1000,
        num_seeds: int = 1, 
        num_workers: int = 1, 
        jobs_per_gpu: int = 1, 
        use_gpu: bool = True,
        wandb_project="genagg",
        single_thread=True,
        save=False,
        ):

    m = multiprocessing.Manager()
    q = m.Queue()
    seeds = torch.randint(torch.iinfo(torch.int).max, size=(num_seeds,))

    if use_gpu:
        num_gpus = int(num_workers // jobs_per_gpu)
        gpu_semaphores = list(itertools.chain(list(range(num_gpus)) * jobs_per_gpu))
        [q.put(s) for s in gpu_semaphores]

    exps = []
    for seed in seeds:
        for dataset in datasets:
            for name, config in configs.items():
                if single_thread:
                    run_exp(name, config, dataset, q, use_gpu, seed, wandb_project, batch_size, epochs, save)
                else:
                    exps.append(
                        (
                            name, config, dataset, q, use_gpu, seed, wandb_project,
                            batch_size, epochs, save
                        )
                    )

    if not single_thread:
        with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
            pool.map(run_exp_args, exps)

