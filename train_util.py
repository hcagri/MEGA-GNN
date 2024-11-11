import torch
import tqdm
from torch_geometric.transforms import BaseTransform
from typing import Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score
import json
import os
from data_util import assign_ports_with_cpp, create_hetero_obj
import numpy as np
import sklearn.metrics
import negative_sampling

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
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        else:
            nodes = torch.unique(data['node', 'to', 'node'].edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else: 
            data['node'].x = torch.cat([x, ids], dim=1)
        
        return data

def extract_param(parameter_name: str, args) -> float:
    
    """
    Extract the value of the specified parameter for the given model.
    
    Args:
    - parameter_name (str): Name of the parameter (e.g., "lr").
    - args (argparser): Arguments given to this specific run.
    
    Returns:
    - float: Value of the specified parameter.
    """
    if 'ETH' in args.data:
        file_path = './model_settings_ETH.json'
    else:
        file_path = './model_settings.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    return data.get(args.model, {}).get("params", {}).get(parameter_name, None)

def add_arange_ids(data_list):
    '''
    Add the index as an id to the edge features to find seed edges in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    '''
    for data in data_list:
        if isinstance(data, HeteroData):
            data['node', 'to', 'node'].edge_attr = torch.cat([torch.arange(data['node', 'to', 'node'].edge_attr.shape[0]).view(-1, 1), data['node', 'to', 'node'].edge_attr], dim=1)
            offset = data['node', 'to', 'node'].edge_attr.shape[0]
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([torch.arange(offset, data['node', 'rev_to', 'node'].edge_attr.shape[0] + offset).view(-1, 1), data['node', 'rev_to', 'node'].edge_attr], dim=1)
        else:
            data.edge_attr = torch.cat([torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)

def get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args):
    if isinstance(tr_data, HeteroData):
        tr_edge_label_index = tr_data['node', 'to', 'node'].edge_index
        tr_edge_label = tr_data['node', 'to', 'node'].y


        tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), tr_edge_label_index), 
                                    edge_label=tr_edge_label, batch_size=args.batch_size, shuffle=True, transform=transform)
        
        val_edge_label_index = val_data['node', 'to', 'node'].edge_index[:,val_inds]
        val_edge_label = val_data['node', 'to', 'node'].y[val_inds]


        val_loader =  LinkNeighborLoader(val_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), val_edge_label_index), 
                                    edge_label=val_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform)
        
        te_edge_label_index = te_data['node', 'to', 'node'].edge_index[:,te_inds]
        te_edge_label = te_data['node', 'to', 'node'].y[te_inds]


        te_loader =  LinkNeighborLoader(te_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), te_edge_label_index), 
                                    edge_label=te_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform)
    else:
        tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, batch_size=args.batch_size, shuffle=True, transform=transform)
        val_loader = LinkNeighborLoader(val_data,num_neighbors=args.num_neighs, edge_label_index=val_data.edge_index[:, val_inds],
                                        edge_label=val_data.y[val_inds], batch_size=args.batch_size, shuffle=False, transform=transform)
        te_loader =  LinkNeighborLoader(te_data,num_neighbors=args.num_neighs, edge_label_index=te_data.edge_index[:, te_inds],
                                edge_label=te_data.y[te_inds], batch_size=args.batch_size, shuffle=False, transform=transform)
        
    return tr_loader, val_loader, te_loader

@torch.no_grad()
def evaluate_homo(loader, inds, model, data, device, args):
    model.eval()
    assert not model.training, "Test error: Model is not in evaluation mode"
    '''Evaluates the model performane for homogenous graph data.'''
    preds = []
    ground_truths = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch.n_id
            add_edge_index = data.edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data.edge_attr[missing_ids, :].detach().clone()
            add_y = data.y[missing_ids].detach().clone()
        
            batch.edge_index = torch.cat((batch.edge_index, add_edge_index), 1)
            batch.edge_attr = torch.cat((batch.edge_attr, add_edge_attr), 0)
            batch.y = torch.cat((batch.y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        #remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]
        
        with torch.no_grad():
            batch.to(device)

            out = model(batch)
            out = out[mask]
            pred = out
            preds.append(pred.detach().cpu())
            ground_truths.append(batch.y[mask].detach().cpu())
            
    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()

    # Compute Metrics
    f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)
    model.train()
    return f1, auc, precision, recall

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
            # To be consistent, sample the edges for forward and backward edge types.
            assign_ports_with_cpp(batch) 
    
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
        batch_edge_ids = loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            # Just ignore this part we rae not entering here args.data == "Small_HI"
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch['node'].n_id
            add_edge_index = data['node', 'to', 'node'].edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data['node', 'to', 'node'].edge_attr[missing_ids, :].detach().clone()
            add_y = data['node', 'to', 'node'].y[missing_ids].detach().clone()
        
            batch['node', 'to', 'node'].edge_index = torch.cat((batch['node', 'to', 'node'].edge_index, add_edge_index), 1)
            batch['node', 'to', 'node'].edge_attr = torch.cat((batch['node', 'to', 'node'].edge_attr, add_edge_attr), 0)
            batch['node', 'to', 'node'].y = torch.cat((batch['node', 'to', 'node'].y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        #remove the unique edge id from the edge features, as it's no longer needed
        batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
        batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
        
        with torch.no_grad():
            batch.to(device)

            out = model(batch)
                
            out = out[mask]
            pred = out

            preds.append(pred.detach().cpu())
            ground_truths.append(batch['node', 'to', 'node'].y[mask].detach().cpu())

    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()

    # Compute Metrics
    f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)


    model.train()
    return f1, auc, precision, recall

@torch.no_grad()
def evaluate_hetero_lp(loader, inds, model, data, device, args, mode='eval'):
    '''Evaluates the model performane for heterogenous graph data.'''
    model.eval()
    assert not model.training, "Test error: Model is not in evaluation mode"

    eval_preds_labels = {"pos_pred": [], "neg_pred": [], "pos_labels": [], "neg_labels": []}
    batch_metrics = { 'lp_mean_acc': [], 'lp_pos_f1': [], 'lp_auc': [], 'lp_mrr': [], 'lp_hits@1': [], 
                     'lp_hits@2': [], 'lp_hits@5': [], 'lp_hits@10': []}
    

    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #remove the unique edge id from the edge features, as it's no longer needed
        batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
        batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

        ind_mask = batch['node', 'to', 'node'].e_id > inds[0]
        negative_edge_sampling(batch, args, ind_mask)

        if args.edge_agg_type == 'adamm':
            batch = HeteroToMultigraph(batch)

        if args.ports and args.ports_batch:
            # To be consistent, sample the edges for forward and backward edge types.
            assign_ports_with_cpp(batch) 
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch)

            if isinstance(batch, HeteroData):
                pos_labels = batch['node', 'to', 'node'].pos_y
                pos_pred = out[0]
                neg_labels = batch['node', 'to', 'node'].neg_y
                neg_pred = out[1]
            else:
                pos_labels = batch.pos_y
                pos_pred = out[0]
                neg_labels = batch.neg_y
                neg_pred = out[1]     
            # Free some GPU memory
            pos_pred, neg_pred, pos_labels, neg_labels = pos_pred.detach().cpu(), neg_pred.detach().cpu(), pos_labels.detach().cpu(), neg_labels.detach().cpu()
            
            # GNN predictions and metrics (link prediction)
            gnn_mrr, gnn_hits_dict = compute_mrr(pos_pred, neg_pred, [1,2,5,10])
            gnn_auc = compute_auc(pos_pred, neg_pred, pos_labels, neg_labels)
            for key in gnn_hits_dict:
                batch_metrics['lp_'+key].append(gnn_hits_dict[key])
            batch_metrics['lp_mrr'].append(gnn_mrr)
            batch_metrics['lp_auc'].append(gnn_auc)
            gnn_preds_labels = gnn_get_predictions_and_labels(pos_pred, neg_pred, pos_labels, neg_labels)
            lp_metrics = lp_compute_metrics(**gnn_preds_labels)
            for key in lp_metrics:
                batch_metrics[key].append(lp_metrics[key])
            for k, v in gnn_preds_labels.items(): 
                eval_preds_labels[k].extend(v.tolist())
        

    ## Compute eval metrics
    if mode == 'eval':
        eval_metrics = {f'ev_{met}': np.mean(batch_metrics[met]) for met in ['lp_mean_acc', 'lp_auc', 'lp_mrr', 'lp_hits@1', 'lp_hits@2', 'lp_hits@5', 'lp_hits@10']}
        eval_preds_labels = {k: np.array(v) for k, v in eval_preds_labels.items()}
        lp_metrics = lp_compute_metrics(**gnn_preds_labels)
        for k, v in lp_metrics.items():
            eval_metrics['ev_' + k] = v
    elif mode == 'test':
        eval_metrics = {f'te_{met}': np.mean(batch_metrics[met]) for met in ['lp_mean_acc', 'lp_auc', 'lp_mrr', 'lp_hits@1', 'lp_hits@2', 'lp_hits@5', 'lp_hits@10']}
        eval_preds_labels = {k: np.array(v) for k, v in eval_preds_labels.items()}
        lp_metrics = lp_compute_metrics(**gnn_preds_labels)
        for k, v in lp_metrics.items():
            eval_metrics['te_' + k] = v

    model.train()
    return eval_metrics

@torch.no_grad()
def evaluate_homo_lp(loader, inds, model, data, device, args, mode='eval'):
    '''Evaluates the model performane for heterogenous graph data.'''
    model.eval()
    assert not model.training, "Test error: Model is not in evaluation mode"

    eval_preds_labels = {"pos_pred": [], "neg_pred": [], "pos_labels": [], "neg_labels": []}
    batch_metrics = { 'lp_mean_acc': [], 'lp_pos_f1': [], 'lp_auc': [], 'lp_mrr': [], 'lp_hits@1': [], 
                     'lp_hits@2': [], 'lp_hits@5': [], 'lp_hits@10': []}
    

    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]

        ind_mask = batch.e_id > inds[0]
        negative_edge_sampling(batch, args, ind_mask)

        if args.edge_agg_type=='adamm':
            batch = ToMultigraph(batch)
        else:
            batch = create_hetero_obj(batch.x, batch.y, batch.edge_index, batch.edge_attr, batch.timestamps, args, batch.simp_edge_batch, batch)


        if args.ports and args.ports_batch:
            # To be consistent, sample the edges for forward and backward edge types.
            assign_ports_with_cpp(batch) 
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch)

            if isinstance(batch, HeteroData):
                pos_labels = batch['node', 'to', 'node'].pos_y
                pos_pred = out[0]
                neg_labels = batch['node', 'to', 'node'].neg_y
                neg_pred = out[1]
            else:
                pos_labels = batch.pos_y
                pos_pred = out[0]
                neg_labels = batch.neg_y
                neg_pred = out[1]

            # Free some GPU memory
            pos_pred, neg_pred, pos_labels, neg_labels = pos_pred.detach().cpu(), neg_pred.detach().cpu(), pos_labels.detach().cpu(), neg_labels.detach().cpu()
            
            # GNN predictions and metrics (link prediction)
            gnn_mrr, gnn_hits_dict = compute_mrr(pos_pred, neg_pred, [1,2,5,10])
            gnn_auc = compute_auc(pos_pred, neg_pred, pos_labels, neg_labels)
            for key in gnn_hits_dict:
                batch_metrics['lp_'+key].append(gnn_hits_dict[key])
            batch_metrics['lp_mrr'].append(gnn_mrr)
            batch_metrics['lp_auc'].append(gnn_auc)
            gnn_preds_labels = gnn_get_predictions_and_labels(pos_pred, neg_pred, pos_labels, neg_labels)
            lp_metrics = lp_compute_metrics(**gnn_preds_labels)
            for key in lp_metrics:
                batch_metrics[key].append(lp_metrics[key])
            for k, v in gnn_preds_labels.items(): 
                eval_preds_labels[k].extend(v.tolist())
        

    ## Compute eval metrics
    if mode == 'eval':
        eval_metrics = {f'ev_{met}': np.mean(batch_metrics[met]) for met in ['lp_mean_acc', 'lp_auc', 'lp_mrr', 'lp_hits@1', 'lp_hits@2', 'lp_hits@5', 'lp_hits@10']}
        eval_preds_labels = {k: np.array(v) for k, v in eval_preds_labels.items()}
        lp_metrics = lp_compute_metrics(**gnn_preds_labels)
        for k, v in lp_metrics.items():
            eval_metrics['ev_' + k] = v
    elif mode == 'test':
        eval_metrics = {f'te_{met}': np.mean(batch_metrics[met]) for met in ['lp_mean_acc', 'lp_auc', 'lp_mrr', 'lp_hits@1', 'lp_hits@2', 'lp_hits@5', 'lp_hits@10']}
        eval_preds_labels = {k: np.array(v) for k, v in eval_preds_labels.items()}
        lp_metrics = lp_compute_metrics(**gnn_preds_labels)
        for k, v in lp_metrics.items():
            eval_metrics['te_' + k] = v

    model.train()
    return eval_metrics


def save_model(model, optimizer, epoch, args, data_config):
    # Save the model in a dictionary
    if not os.path.exists(os.path.join(f'{data_config["paths"]["model_to_save"]}', args.unique_name)):
        os.mkdir(os.path.join(f'{data_config["paths"]["model_to_save"]}', args.unique_name))

    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, 
                os.path.join(f'{data_config["paths"]["model_to_save"]}', args.unique_name, f'epoch_{epoch+1}.tar')
            )
    
def load_model(model, device, args, config, data_config):
    checkpoint = torch.load(f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def negative_edge_sampling(batch, args, inds_mask=None):

    '''
        Sample positive and negative edges from given subgraph for link prediction objective
        
        Args:
            batch: Sampled Subgraph
    '''
    if isinstance(batch, HeteroData):
        #1. add the negative samples
        E = batch['node', 'to', 'node'].edge_index.shape[1]
        
        positions = torch.arange(E)
        if inds_mask is not None:
            # for validation and test only select the positive edges from validation or test graph edges. Do not select training edges as positive edges in evaluation.
            positions = positions[inds_mask] 

        drop_count = min(args.batch_size, int(len(positions) * 0.15)) # 15% probability to drop an edge or maximally args.batch_size edges
        if len(positions) > 0 and drop_count > 0:
            drop_idxs = torch.multinomial(torch.full((len(positions),), 1.0), drop_count, replacement=False) #[drop_count, ]
        else:
            drop_idxs = torch.tensor([]).long()
        drop_positions = positions[drop_idxs]

        mask = torch.zeros((E,)).long() #[E, ]
        mask = mask.index_fill_(dim=0, index=drop_positions, value=1).bool() #[E, ]

        input_edge_index = batch['node', 'to', 'node'].edge_index[:, ~mask]
        input_edge_attr  = batch['node', 'to', 'node'].edge_attr[~mask]

        pos_edge_index = batch['node', 'to', 'node'].edge_index[:, mask]
        pos_edge_attr  = batch['node', 'to', 'node'].edge_attr[mask]

        # Discard the reverse of the possitive edges if any in the 'rev_to'
        pos_edges_with_feats = torch.cat([pos_edge_index.T, pos_edge_attr], dim=1)
        reverse_combined = torch.cat([
            batch['node', 'rev_to', 'node'].edge_index[[1, 0], :].T, # Edge indices are reverse becase positive edges are the original edges, we would like to discard reverse ones.
            batch['node', 'rev_to', 'node'].edge_attr
            ], 
            dim=1)
        
        _, idx, counts = torch.cat([pos_edges_with_feats, reverse_combined], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
        mask_all = torch.isin(idx, torch.where(counts.gt(1))[0])
        mask_rev = mask_all[len(pos_edges_with_feats):]  # tensor([ True, False, False,  True], device='cuda:0')
        
         # Sample Negative Edges
        neg_edges_src, neg_edges_dst = negative_sampling.generate_negative_samples(batch['node', 'to', 'node'].edge_index.tolist(), pos_edge_index.tolist(), 64)
        
        neg_edge_index = torch.stack([torch.tensor(neg_edges_src), torch.tensor(neg_edges_dst)], dim=0)
        neg_edge_attr = pos_edge_attr.repeat_interleave(64,dim=0)

        # Update the batch object
        batch['node', 'to', 'node'].edge_index, batch['node', 'to', 'node'].edge_attr = input_edge_index, input_edge_attr
        batch['node', 'to', 'node'].pos_edge_index, batch['node', 'to', 'node'].pos_edge_attr = pos_edge_index, pos_edge_attr
        batch['node', 'to', 'node'].neg_edge_index, batch['node', 'to', 'node'].neg_edge_attr = neg_edge_index, neg_edge_attr
        batch['node', 'to', 'node'].timestamps = batch['node', 'to', 'node'].timestamps[~mask]

        batch['node', 'to', 'node'].pos_y = torch.ones(pos_edge_index.shape[1]  , dtype=torch.int32)
        batch['node', 'to', 'node'].neg_y = torch.zeros(neg_edge_index.shape[1], dtype=torch.int32)

        batch['node', 'rev_to', 'node'].edge_index = batch['node', 'rev_to', 'node'].edge_index[:, ~mask_rev]
        batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[~mask_rev, :]
        batch['node', 'rev_to', 'node'].timestamps = batch['node', 'rev_to', 'node'].timestamps[~mask_rev]
        batch['node', 'rev_to', 'node'].e_id = batch['node', 'rev_to', 'node'].e_id[~mask_rev]

        if args.flatten_edges:
            batch['node', 'to', 'node'].simp_edge_batch = batch['node', 'to', 'node'].simp_edge_batch[~mask]
            batch['node', 'rev_to', 'node'].simp_edge_batch = batch['node', 'rev_to', 'node'].simp_edge_batch[~mask_rev]

    else:
        #1. add the negative samples
        E = batch.edge_index.shape[1]
        
        positions = torch.arange(E)
        if inds_mask is not None:
            # for validation and test only select the positive edges from validation or test graph edges. Do not select training edges as positive edges in evaluation.
            positions = positions[inds_mask] 

        drop_count = min(args.batch_size, int(len(positions) * 0.15)) # 15% probability to drop an edge or maximally args.batch_size edges
        if len(positions) > 0 and drop_count > 0:
            drop_idxs = torch.multinomial(torch.full((len(positions),), 1.0), drop_count, replacement=False) #[drop_count, ]
        else:
            drop_idxs = torch.tensor([]).long()
        drop_positions = positions[drop_idxs]

        mask = torch.zeros((E,)).long() #[E, ]
        mask = mask.index_fill_(dim=0, index=drop_positions, value=1).bool() #[E, ]

        input_edge_index = batch.edge_index[:, ~mask]
        input_edge_attr  = batch.edge_attr[~mask]

        pos_edge_index = batch.edge_index[:, mask]
        pos_edge_attr  = batch.edge_attr[mask]
        
        # Sample Negative Edges
        neg_edges_src, neg_edges_dst = negative_sampling.generate_negative_samples(batch.edge_index.tolist(), pos_edge_index.tolist(), 64)
        
        neg_edge_index = torch.stack([torch.tensor(neg_edges_src), torch.tensor(neg_edges_dst)], dim=0)
        neg_edge_attr = pos_edge_attr.repeat_interleave(64,dim=0)

        # Update the batch object
        batch.edge_index, batch.edge_attr = input_edge_index, input_edge_attr
        batch.pos_edge_index, batch.pos_edge_attr = pos_edge_index, pos_edge_attr
        batch.neg_edge_index, batch.neg_edge_attr = neg_edge_index, neg_edge_attr
        batch.timestamps = batch.timestamps[~mask]

        batch.pos_y = torch.ones(pos_edge_index.shape[1]  , dtype=torch.int32)
        batch.neg_y = torch.zeros(neg_edge_index.shape[1], dtype=torch.int32)

        if args.flatten_edges:
            batch.simp_edge_batch = batch.simp_edge_batch[~mask]
    return


def compute_mrr(pos_pred, neg_pred, ks):
    """Compute mean reciprocal rank (MRR) and Hits@k for link prediction.
    
    Returns
    -------
    float, dict[str, float]
        MRR and dictionnary with Hits@k metrics. 
    """
    pos_pred = pos_pred.detach().clone().cpu().numpy().flatten()
    neg_pred = neg_pred.detach().clone().cpu().numpy().flatten()

    num_positives = len(pos_pred)
    neg_pred_reshaped = neg_pred.reshape(num_positives, 64)

    mrr_scores = []
    keys = [f'hits@{k}' for k in ks]
    hits_dict = {key: 0 for key in keys}
    count = 0

    for pos, neg in zip(pos_pred, neg_pred_reshaped):
        # Combine positive and negative predictions
        combined = np.concatenate([neg, [pos]])  # Add positive prediction to the end

        # Rank predictions (argsort twice gives the ranks)
        ranks = (-combined).argsort().argsort() + 1  # Add 1 because ranks start from 1
        for k, key in zip(ks, keys):
            if ranks[-1] <= k:
                hits_dict[key] += 1
        
        count += 1
        # Reciprocal rank of positive prediction (which is the last one in combined)
        reciprocal_rank = 1 / ranks[-1]
        mrr_scores.append(reciprocal_rank)
    
    # Calculate Hits@k
    for key in keys:
        hits_dict[key] /= count

    # Calculate Mean Reciprocal Rank
    mrr = np.mean(mrr_scores)
    
    return mrr, hits_dict


def compute_auc(pos_probs, neg_probs, pos_labels, neg_labels):
    """Compute the area under the curve (AUC) of precision/recall for link prediction."""
    probs = np.concatenate((pos_probs, neg_probs), axis=0)
    labels = np.concatenate((pos_labels, neg_labels), axis=0)

    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, probs, pos_label=1)
    auc = sklearn.metrics.auc(recall, precision)

    return auc


def gnn_get_predictions_and_labels(pos_pred, neg_pred, pos_label, neg_label) -> dict[str, np.ndarray]:
    pos_pred = (pos_pred >= 0.5).float()
    neg_pred = (neg_pred >= 0.5).float()

    # All the outputs are numpy ndarrays
    return {
        "pos_pred": pos_pred.detach().clone().cpu().numpy().flatten(), "neg_pred": neg_pred.detach().clone().cpu().numpy().flatten(),
        "pos_labels": pos_label.detach().clone().cpu().numpy(), "neg_labels": neg_label.detach().clone().cpu().numpy()
    }

def lp_compute_metrics(pos_pred, neg_pred, pos_labels, neg_labels) -> dict[str, float]:
    """Compute mean accuracy and positive F1 score for link prediction."""
    # Calculate positive accuracy
    pos_accuracy = np.mean(pos_pred == pos_labels)
    
    # Calculate negative accuracy
    neg_accuracy = np.mean(neg_pred == neg_labels)
    
    # Calculate overall accuracy
    mean_accuracy = (pos_accuracy + neg_accuracy) / 2

    # Calculate TP, FP, FN
    preds = np.concatenate((pos_pred, neg_pred), axis=0)
    labels = np.concatenate((pos_labels, neg_labels), axis=0)
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    # Calculate Precision and Recall for positive class
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate positive F1 Score
    pos_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return { 'lp_mean_acc': mean_accuracy, 'lp_pos_f1': pos_f1 }


def compute_binary_metrics(preds: np.array, labels: np.array):
    """
    Computes metrics based on raw/ normalized model predictions
    :param preds: Raw (or normalized) predictions (can vary threshold here if raw scores are provided)
    :param labels: Binary target labels
    :return: Accuracy, illicit precision/ recall/ F1, and ROC AUC scores
    """
    probs = preds[:,1]
    preds = preds.argmax(axis=-1)

    precisions, recalls, _ = sklearn.metrics.precision_recall_curve(labels, probs) # probs: probabilities for the positive class
    f1 = sklearn.metrics.f1_score(labels, preds, zero_division=0)
    auc = sklearn.metrics.auc(recalls, precisions)

    precision = sklearn.metrics.precision_score(labels, preds, zero_division=0)
    recall = sklearn.metrics.recall_score(labels, preds, zero_division=0)

    return f1, auc, precision, recall



def ToMultigraph(data): 
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    
    # create full edge_index, assume that original input edge_index is single direction directed, mult-edge is allowed
    self_loop_indice = edge_index[0] == edge_index[1]
    self_loops = edge_index[:, self_loop_indice]
    other_edges = edge_index[:, ~self_loop_indice]
    reversed_other_edges = torch.stack([other_edges[1], other_edges[0]])

    edge_index = torch.cat([self_loops, other_edges, reversed_other_edges], dim=-1)
    if edge_attr is not None:
        edge_attr = torch.cat([edge_attr[self_loop_indice], edge_attr[~self_loop_indice], edge_attr[~self_loop_indice]], dim=0)
    edge_direction = torch.cat([torch.full((self_loops.size(-1),), 0), torch.full((other_edges.size(-1),), 1), torch.full((reversed_other_edges.size(-1),), 2)], dim=0)
    
    # map to simplified edge, currently ignore the edge direction (this is fine)
    simplified_edge_mapping = {}
    simplified_edge_batch = []
    i = 0
    for edge in edge_index.T:
        # transform edge to tuple
        tuple_edge = tuple(edge.tolist())
        if tuple_edge not in simplified_edge_mapping:
            simplified_edge_mapping[tuple_edge] = i

            # simplified_edge_index.append(edge)
            # simplified_edge_mapping[tuple_edge[::-1]] = i+1 #
            i += 1
        simplified_edge_batch.append(simplified_edge_mapping[tuple_edge])
    simplified_edge_batch = torch.LongTensor(simplified_edge_batch)

    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.edge_direction = edge_direction
    data.simp_edge_batch = simplified_edge_batch
    return data

def HeteroToMultigraph(data): 
    x, edge_index, edge_attr = data['node'].x, data['node', 'to', 'node'].edge_index, data['node', 'to', 'node'].edge_attr
    r_edge_index, r_edge_attr =  data['node', 'rev_to', 'node'].edge_index, data['node', 'rev_to', 'node'].edge_attr

    # create full edge_index, assume that original input edge_index is single direction directed, mult-edge is allowed
    self_loop_indice = edge_index[0] == edge_index[1]
    r_self_loop_indice = r_edge_index[0] == r_edge_index[1]

    self_loops = edge_index[:, self_loop_indice]
    r_self_loops = r_edge_index[:, r_self_loop_indice]

    other_edges = edge_index[:, ~self_loop_indice]
    r_other_edges = r_edge_index[:, ~r_self_loop_indice]

    edge_index = torch.cat([self_loops, r_self_loops, other_edges, r_other_edges], dim=-1)
    if edge_attr is not None:
        edge_attr = torch.cat([edge_attr[self_loop_indice], r_edge_attr[r_self_loop_indice], edge_attr[~self_loop_indice], r_edge_attr[~r_self_loop_indice]], dim=0)

    edge_direction = torch.cat([
        torch.full((self_loops.size(-1),), 0), 
        torch.full((r_self_loops.size(-1),), 0), 
        torch.full((other_edges.size(-1),), 1), 
        torch.full((r_other_edges.size(-1),), 2)
        ], dim=0)
    
    # map to simplified edge, currently ignore the edge direction (this is fine)
    simplified_edge_mapping = {}
    simplified_edge_batch = []
    i = 0
    for edge in edge_index.T:
        # transform edge to tuple
        tuple_edge = tuple(edge.tolist())
        if tuple_edge not in simplified_edge_mapping:
            simplified_edge_mapping[tuple_edge] = i

            # simplified_edge_index.append(edge)
            # simplified_edge_mapping[tuple_edge[::-1]] = i+1 #
            i += 1
        simplified_edge_batch.append(simplified_edge_mapping[tuple_edge])
    simplified_edge_batch = torch.LongTensor(simplified_edge_batch)

    new_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_direction = edge_direction, simp_edge_batch = simplified_edge_batch)

    for key in ['pos_edge_index', 'pos_edge_attr', 'neg_edge_index', 'neg_edge_attr', 'pos_y', 'neg_y']:
        new_data[key] = data['node', 'to', 'node'][key]
    
    return new_data
