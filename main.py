import time
import logging
from util import create_parser, set_seed, logger_setup
from data_loading import get_data, get_eth_data, get_eth_kaggle_data
from training import train_gnn
from training_eth import train_gnn_eth
from inference import infer_gnn
from train_util import extract_param
import json
import wandb 
import random
import os

def main():
    parser = create_parser()
    args = parser.parse_args()

    args.num_neighs = [int(t) for t in args.num_neighs]

    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)

    # Setup logging
    logger_setup()

    # Check Argument consistency
    if args.task == 'edge_class':
        assert args.data in ['Small_HI', 'Small_LI', 'Medium_HI', 'Medium_LI']
    elif args.task == 'node_class':
        assert args.data in ['ETH', 'ETH-Kaggle']
    if args.edge_agg_type == 'adamm':
        assert args.reverse_mp == False
        assert args.task == 'node_class' or args.task == 'lp'

    #set seed
    if args.seed == 1:
        args.seed = random.randint(2, 256000)

    set_seed(args.seed)
    if args.flatten_edges:
        name = f"{args.data} | {args.model}+EdgeAgg={args.edge_agg_type}"
    else:
        name = f"{args.data} | Multi-{args.model}"
        if args.ports_batch:
            name += " Ports Batch"
    if args.node_agg_type == 'genagg':
        name += " NodeAgg=GenAgg"
    
    if args.adamm_hetero:
        name+="adam-hetero"

    project = "ICLR-LP" if args.task =='lp' else "ICLR"

    #define a model config dictionary and wandb logging at the same time
    wandb.init(
        mode="disabled" if args.testing else "online",
        project=project, #replace this with your wandb project name if you want to use wandb logging
        entity="hcbilgi",
        name=name,
        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )

    wandb.run.log_code(
                ".",
                include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml") or path.endswith(".sh"),
                exclude_fn=lambda path, root: os.path.relpath(path, root).startswith("cache/"),
            )
            
    args.unique_name = wandb.run.dir.split('/')[-2].split('-')[-1]

    # Print the config 
    print("----- CONFIG -----")
    for key, value in wandb.config.items():
        print(key, ':', value)
    print(2*"------------------")

    print("----- ARGS -----")
    for key, value in vars(args).items():
        print(key, ':', value)
    print(2*"------------------")

    #get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()

    if (args.data != "ETH") and ((args.data != "ETH-Kaggle")):
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    elif args.data == "ETH-Kaggle":
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_eth_kaggle_data(args, data_config)
    else:
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_eth_data(args, data_config)
    
    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    if not args.inference:
        #Training
        logging.info(f"Running Training")
        if args.data != "ETH" and args.data != "ETH-Kaggle":
            train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)
        else:
            train_gnn_eth(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)
    else:
        #Inference
        logging.info(f"Running Inference")
        infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)

if __name__ == "__main__":
    main()
