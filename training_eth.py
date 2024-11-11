import torch
from typing import Union
import tqdm
from sklearn.metrics import f1_score
from train_util import extract_param, save_model, load_model, negative_edge_sampling, compute_binary_metrics, ToMultigraph
from training import train_hetero_lp
from data_util import z_norm, assign_ports_with_cpp
from models import MultiMPNN
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
from torch_geometric.loader import NeighborLoader
import wandb
import logging
import os

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    
    for epoch in range(config.epochs):
        logging.info(f"---------- Epoch {epoch} ----------")
        
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        
        assert model.training, "Training error: Model is not in training mode"

        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            ''' Add port numberings after neighborhood sampling. ''' 
            if args.ports and args.ports_batch:
                # To be consistent, sample the edges for forward and backward edge types.
                assign_ports_with_cpp(batch) 

            if args.edge_agg_type=='adamm':
                batch = ToMultigraph(batch)
            
            optimizer.zero_grad()

            # select the seed nodes (previously edges) from which the batch was created
            # This will correspond to the first batch_size nodes, which are the seed nodes.
            inds = tr_inds.detach().cpu()
            batch_node_inds = inds[batch.input_id.detach().cpu()]
            batch_node_ids = tr_loader.data.x.detach().cpu()[batch_node_inds, 0]
            mask = torch.isin(batch.x[:, 0].detach().cpu(), batch_node_ids)
            
            #remove the unique node id from the node features, as it's no longer needed
            batch.x = batch.x[:, 1:]

            batch.to(device)

            out = model(batch)

            pred = out[mask]
            ground_truth = batch.y[mask]
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

            preds.append(pred.detach().cpu())
            ground_truths.append(ground_truth.detach().cpu())

            
        pred = torch.cat(preds, dim=0).numpy()
        ground_truth = torch.cat(ground_truths, dim=0).numpy()

        f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)

        wandb.log({"f1/train": f1}, step=epoch)
        wandb.log({"precision/train": precision}, step=epoch)
        wandb.log({"recall/train": recall}, step=epoch)
        wandb.log({"auc/train": auc}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')
        logging.info(f'Train Precision: {precision:.4f}')
        logging.info(f'Train Recall: {recall:.4f}')
        logging.info(f'Train Auc: {auc:.4f}')

        #evaluate
        val_f1, val_auc, val_precision, val_recall = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        te_f1, te_auc, te_precision, te_recall = evaluate_homo(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"precision/validation": val_precision}, step=epoch)
        wandb.log({"recall/validation": val_recall}, step=epoch)
        wandb.log({"auc/validation": val_auc}, step=epoch)
        logging.info(f'Val F1: {val_f1:.4f}')
        logging.info(f'Val Precision: {val_precision:.4f}')
        logging.info(f'Val Recall: {val_recall:.4f}')
        logging.info(f'Val Auc: {val_auc:.4f}')

        wandb.log({"f1/test": te_f1}, step=epoch)
        wandb.log({"precision/test": te_precision}, step=epoch)
        wandb.log({"recall/test": te_recall}, step=epoch)
        wandb.log({"auc/test": te_auc}, step=epoch)
        logging.info(f'Test F1: {te_f1:.4f}')
        logging.info(f'Test Precision: {te_precision:.4f}')
        logging.info(f'Test Recall: {te_recall:.4f}')
        logging.info(f'Test Auc: {te_auc:.4f}')

        wandb.log({"Loss": total_loss/total_examples}, step=epoch)
        
        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
        
    return model



def train_hetero_eth(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        logging.info(f"---------- Epoch {epoch} ----------")
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        
        assert model.training, "Training error: Model is not in training mode"

        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            
            ''' Add port numberings after neighborhood sampling. ''' 
            if args.ports and args.ports_batch:
                # To be consistent, sample the edges for forward and backward edge types.
                assign_ports_with_cpp(batch) 
            
            optimizer.zero_grad()

            # select the seed nodes (previously edges) from which the batch was created
            # This will correspond to the first batch_size nodes, which are the seed nodes.
            inds = tr_inds.detach().cpu()
            batch_node_inds = inds[batch['node'].input_id.detach().cpu()]
            batch_node_ids = tr_loader.data['node'].x.detach().cpu()[batch_node_inds, 0]
            mask = torch.isin(batch['node'].x[:, 0].detach().cpu(), batch_node_ids)
            
            #remove the unique node id from the node features, as it's no longer needed
            batch['node'].x = batch['node'].x[:, 1:]

            batch.to(device)

            out = model(batch)

            pred = out[mask]
            ground_truth = batch['node'].y[mask]
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

            preds.append(pred.detach().cpu())
            ground_truths.append(ground_truth.detach().cpu())
            
        pred = torch.cat(preds, dim=0).numpy()
        ground_truth = torch.cat(ground_truths, dim=0).numpy()

        f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)

        wandb.log({"f1/train": f1}, step=epoch)
        wandb.log({"precision/train": precision}, step=epoch)
        wandb.log({"recall/train": recall}, step=epoch)
        wandb.log({"auc/train": auc}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')
        logging.info(f'Train Precision: {precision:.4f}')
        logging.info(f'Train Recall: {recall:.4f}')
        logging.info(f'Train Auc: {auc:.4f}')

        #evaluate
        val_f1, val_auc, val_precision, val_recall = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
        te_f1, te_auc, te_precision, te_recall = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"precision/validation": val_precision}, step=epoch)
        wandb.log({"recall/validation": val_recall}, step=epoch)
        wandb.log({"auc/validation": val_auc}, step=epoch)
        logging.info(f'Val F1: {val_f1:.4f}')
        logging.info(f'Val Precision: {val_precision:.4f}')
        logging.info(f'Val Recall: {val_recall:.4f}')
        logging.info(f'Val Auc: {val_auc:.4f}')

        wandb.log({"f1/test": te_f1}, step=epoch)
        wandb.log({"precision/test": te_precision}, step=epoch)
        wandb.log({"recall/test": te_recall}, step=epoch)
        wandb.log({"auc/test": te_auc}, step=epoch)
        logging.info(f'Test F1: {te_f1:.4f}')
        logging.info(f'Test Precision: {te_precision:.4f}')
        logging.info(f'Test Recall: {te_recall:.4f}')
        logging.info(f'Test Auc: {te_auc:.4f}')

        wandb.log({"Loss": total_loss/total_examples}, step=epoch)
        
        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
        
    return model

@torch.no_grad()
def evaluate_hetero(loader, inds, model, data, device, args):
    '''Evaluates the model performane for heterogenous graph data.'''
    model.eval()
    assert not model.training, "Test error: Model is not in evaluation mode"

    preds = []
    ground_truths = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        
        if args.ports and args.ports_batch:
            assign_ports_with_cpp(batch) 
    
        inds = inds.detach().cpu()
        batch_node_inds = inds[batch['node'].input_id.detach().cpu()]
        batch_node_ids = loader.data['node'].x.detach().cpu()[batch_node_inds, 0]
        mask = torch.isin(batch['node'].x[:, 0].detach().cpu(), batch_node_ids)

        #remove the unique node id from the node features, as it's no longer needed
        batch['node'].x = batch['node'].x[:, 1:]
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch)
                
            out = out[mask]
            pred = out
            preds.append(pred.detach().cpu())
            ground_truths.append(batch['node'].y[mask].detach().cpu())

    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()
    f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)

    model.train()
    return f1, auc, precision, recall

@torch.no_grad()
def evaluate_homo(loader, inds, model, data, device, args):
    '''Evaluates the model performane for heterogenous graph data.'''
    model.eval()
    assert not model.training, "Test error: Model is not in evaluation mode"

    preds = []
    ground_truths = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        
        if args.ports and args.ports_batch:
            assign_ports_with_cpp(batch) 
        if args.edge_agg_type=='adamm':
            batch = ToMultigraph(batch)
    
        inds = inds.detach().cpu()
        batch_node_inds = inds[batch.input_id.detach().cpu()]
        batch_node_ids = loader.data.x.detach().cpu()[batch_node_inds, 0]
        mask = torch.isin(batch.x[:, 0].detach().cpu(), batch_node_ids)

        #remove the unique node id from the node features, as it's no longer needed
        batch.x = batch.x[:, 1:]
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch)
                
            out = out[mask]
            pred = out
            preds.append(pred.detach().cpu())
            ground_truths.append(batch.y[mask].detach().cpu())

    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()
    f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)

    model.train()
    return f1, auc, precision, recall


def get_loaders_eth(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args):
    ''' 
        Sampled nodes are sorted based on the order in which they were sampled. In particular, the first batch_size nodes represent 
        the set of original mini-batch nodes.

        In particular, the data loader will add the following attributes to the returned mini-batch:
            `batch_size`        The number of seed nodes (first nodes in the batch)
            `n_id`              The global node index for every sampled node
            `e_id`              The global edge index for every sampled edge
            `input_id`          The global index of the input_nodes
            `num_sampled_nodes` The number of sampled nodes in each hop
            `num_sampled_edges` The number of sampled edges in each hop
    '''
    if isinstance(tr_data, HeteroData):

        tr_loader = NeighborLoader(tr_data, 
                                   num_neighbors= {key: args.num_neighs for key in tr_data.edge_types}, 
                                   batch_size=args.batch_size, 
                                   shuffle=True, 
                                   transform=transform, 
                                   input_nodes=('node', None)
                                   )

        val_loader = NeighborLoader(val_data, 
                                    num_neighbors= {key: args.num_neighs for key in val_data.edge_types}, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    transform=transform,
                                    input_nodes=('node', val_inds),
                                    )
        
        te_loader = NeighborLoader(te_data, 
                                    num_neighbors= {key: args.num_neighs for key in te_data.edge_types}, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    transform=transform,
                                    input_nodes=('node', te_inds),
                                    )
    else:
        tr_loader = NeighborLoader(tr_data, 
                                   num_neighbors= args.num_neighs, 
                                   batch_size=args.batch_size, 
                                   shuffle=True, 
                                   transform=transform
                                   )

        val_loader = NeighborLoader(val_data, 
                                    num_neighbors= args.num_neighs, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    transform=transform,
                                    input_nodes=val_inds,
                                    )
        
        te_loader = NeighborLoader(te_data, 
                                    num_neighbors= args.num_neighs, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    transform=transform,
                                    input_nodes=te_inds,
                                    )
    return tr_loader, val_loader, te_loader


def add_arange_ids(data_list):
    '''
    Add the index as an id to the node features to find seed nodes in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    '''
    for data in data_list:
        if isinstance(data, HeteroData):
            data['node'].x = torch.cat([torch.arange(data['node'].x.shape[0]).view(-1, 1), data['node'].x.view(-1,1)], dim=1)
        else:
            data.x = torch.cat([torch.arange(data.x.shape[0]).view(-1, 1), data.x.view(-1,1)], dim=1)


class AddEgoIds(BaseTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.isin(data.n_id, data.input_id).to(device)
        else:
            nodes = torch.isin(data['node'].n_id, data['node'].input_id).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else: 
            # data['node'].x = torch.cat([x.view(-1, 1), ids], dim=1)
            data['node'].x = torch.cat([x, ids], dim=1)
        
        return data

def get_model(sample_batch, config, args):
    n_feats = (sample_batch.x.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node'].x.shape[1] - 1)
    e_dim = sample_batch.edge_attr.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node', 'to', 'node'].edge_attr.shape[1]
    
    if args.flatten_edges:
        index_ = sample_batch.simp_edge_batch if not isinstance(sample_batch, HeteroData) else sample_batch['node', 'to', 'node'].simp_edge_batch
    else:
        index_ = None
    
    if args.flatten_edges:
         # Instead of in-degree use Fan-in
        if not isinstance(sample_batch, HeteroData):
            s_edges = torch.unique(sample_batch.edge_index, dim=1)
            d = degree(s_edges[1], num_nodes=sample_batch.num_nodes, dtype=torch.long)
        else:
            s_edges = torch.unique(torch.cat((sample_batch['node', 'to', 'node'].edge_index, sample_batch['node', 'rev_to', 'node'].edge_index), 1), dim=1)
            d = degree(s_edges[1], num_nodes=sample_batch.num_nodes, dtype=torch.long)
    else:
        if not isinstance(sample_batch, HeteroData):
            d = degree(sample_batch.edge_index[1], num_nodes=sample_batch.num_nodes, dtype=torch.long)
        else:
            index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
            d = degree(index, num_nodes=sample_batch.num_nodes, dtype=torch.long)

    deg = torch.bincount(d, minlength=1)

    model = MultiMPNN(num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2, 
                    n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim, 
                    final_dropout=config.final_dropout, index_=index_, deg=deg, args=args)   
     
    return model

def train_gnn_eth(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    #set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  
    config = wandb.config

    #set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders_eth(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    #get the model
    sample_batch = next(iter(tr_loader)) 

    if args.ports and args.ports_batch:
        # Add a placeholder for the port features so that the model is loaded correctly!
        if isinstance(sample_batch, HeteroData):
            sample_batch['node', 'to', 'node'].edge_attr = torch.cat([sample_batch['node', 'to', 'node'].edge_attr, torch.zeros((sample_batch['node', 'to', 'node'].edge_attr.shape[0], 2))], dim=1)
            sample_batch['node', 'rev_to', 'node'].edge_attr = torch.cat([sample_batch['node', 'rev_to', 'node'].edge_attr, torch.zeros((sample_batch['node', 'rev_to', 'node'].edge_attr.shape[0], 2))], dim=1)
        else:
            sample_batch.edge_attr = torch.cat([sample_batch.edge_attr, torch.zeros((sample_batch.edge_attr.shape[0], 2))], dim=1)

    model = get_model(sample_batch, config, args)
    
    if args.finetune:
        model, optimizer = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if args.task == 'lp':
        negative_edge_sampling(sample_batch, args)

    if args.edge_agg_type=='adamm':
        sample_batch = ToMultigraph(sample_batch)
    
    sample_batch.to(device)

    if isinstance(sample_batch, HeteroData):
        sample_batch['node'].x = sample_batch['node'].x[:, 1:]
    else:
        sample_batch.x = sample_batch.x[:, 1:]


    logging.info(summary(model, sample_batch))
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    if args.task == 'lp':
        train_hetero_lp(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    if args.reverse_mp:
        model = train_hetero_eth(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    else:
        model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)

    
    wandb.finish()


