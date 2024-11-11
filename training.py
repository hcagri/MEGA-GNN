import torch
import tqdm
from sklearn.metrics import f1_score
import sklearn.metrics
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, negative_edge_sampling, \
    evaluate_hetero, evaluate_hetero_lp, save_model, load_model, compute_mrr, compute_auc, gnn_get_predictions_and_labels, lp_compute_metrics, compute_binary_metrics, \
    ToMultigraph, evaluate_homo_lp, HeteroToMultigraph
from data_util import z_norm, assign_ports_with_cpp, create_hetero_obj
from models import MultiMPNN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
import wandb
import logging
from torch_scatter import scatter
import numpy as np
import os
import time

def lp_loss_fn(input1, input2):
    return -torch.log(input1 + 1e-7).mean() - torch.log(1 - input2 + 1e-7).mean()

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):

            ''' Add port numberings after neighborhood sampling. ''' 
            if args.ports and args.ports_batch:
                # To be consistent, sample the edges for forward and backward edge types.
                assign_ports_with_cpp(batch) 

            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            #remove the unique edge id from the edge features, as it's no longer needed
            batch.edge_attr = batch.edge_attr[:, 1:]

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

def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
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
                assign_ports_with_cpp(batch, process_batch=True) 
            
            optimizer.zero_grad()
            #select the seed edges from which the batch was created

            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            
            #remove the unique edge id from the edge features, as it's no longer needed
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            batch.to(device)

            out = model(batch)
                
            pred = out[mask]
            ground_truth = batch['node', 'to', 'node'].y[mask]
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

def train_hetero_lp(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    global_step = 0
    assert args.task == 'lp', "Training Error: Wrong training script for given task"

    for epoch in range(config.epochs):
        logging.info(f'****** EPOCH {epoch} ******')

        total_loss = total_examples = 0
        batch_metrics = { 'loss': [],
                          'lp_mean_acc': [], 'lp_pos_f1': [], 'lp_auc': [], 
                          'lp_mrr': [], 'lp_hits@1': [], 'lp_hits@2': [], 'lp_hits@5': [], 'lp_hits@10': []
                        }

        epoch_preds_labels = {"pos_pred": [], "neg_pred": [], "pos_labels": [], "neg_labels": []}
        
        assert model.training, "Training error: Model is not in training mode"
        step = 0
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            
            #remove the unique edge id from the edge features, as it's no longer needed
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            negative_edge_sampling(batch, args)

            if args.edge_agg_type == 'adamm':
                batch = HeteroToMultigraph(batch)
            
            ''' Add port numberings after neighborhood sampling. ''' 
            if args.ports and args.ports_batch:
                # To be consistent, sample the edges for forward and backward edge types.
                assign_ports_with_cpp(batch) 
            
            optimizer.zero_grad()

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

            loss = lp_loss_fn(pos_pred, neg_pred)
            batch_metrics['loss'].append(loss.detach().item())

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * (pos_pred.numel() + neg_pred.numel())
            total_examples += (pos_pred.numel() + neg_pred.numel())

            # Free some GPU memory
            pos_pred, neg_pred, pos_labels, neg_labels = pos_pred.detach().cpu(), neg_pred.detach().cpu(), pos_labels.detach().cpu(), neg_labels.detach().cpu()
            loss = loss.detach().cpu()

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
                epoch_preds_labels[k].extend(v.tolist())

            step += 1
            global_step += 1
            # Log batch metrics
            if step % 200 == 0:
                logging.info(f'\nTrain ' + '| '.join([f'{k}: {v[-1]:.4g}' for k,v in batch_metrics.items()]))
                wandb.log({f'batch_{k}': v[-1] for k, v in batch_metrics.items()}, step=global_step)


       ## After epoch ends
        # Log epoch metrics
        epoch_metrics = {f'tr_{met}': np.mean(batch_metrics[met]) for met in ['loss', 'lp_mean_acc', 'lp_auc', 'lp_mrr', 'lp_hits@1', 'lp_hits@2', 'lp_hits@5', 'lp_hits@10']}
        epoch_preds_labels = {k: np.array(v) for k, v in epoch_preds_labels.items()}
        lp_metrics = lp_compute_metrics(**gnn_preds_labels)
        for k, v in lp_metrics.items():
            epoch_metrics['tr_' + k] = v
        logging.info("***** Train results - EPOCH {} *****".format(epoch))
        for k, v in epoch_metrics.items():
            logging.info(f" {k}: {v:.4g}")

        wandb.log(epoch_metrics, step=global_step)

        # Do evaluation
        # Clear CUDA cache before evaluating
        torch.cuda.empty_cache()
        
        val_metrics = evaluate_hetero_lp(val_loader, val_inds, model, val_data, device, args, mode='eval')
        logging.info("***** {} - Eval results *****".format(epoch))
        for key in sorted(val_metrics.keys()):
            logging.info("  %s = %s" % (key, str(val_metrics[key])))

        te_metrics = evaluate_hetero_lp(te_loader, te_inds, model, te_data, device, args, mode='test')
        logging.info("***** {} - Eval results *****".format(epoch))
        for key in sorted(te_metrics.keys()):
            logging.info("  %s = %s" % (key, str(te_metrics[key])))

        # Log eval metrics
        wandb.log(val_metrics, step=global_step)
        wandb.log(te_metrics, step=global_step)

    return model


def train_homo_lp(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    global_step = 0
    assert args.task == 'lp', "Training Error: Wrong training script for given task"

    for epoch in range(config.epochs):
        logging.info(f'****** EPOCH {epoch} ******')

        total_loss = total_examples = 0
        batch_metrics = { 'loss': [],
                          'lp_mean_acc': [], 'lp_pos_f1': [], 'lp_auc': [], 
                          'lp_mrr': [], 'lp_hits@1': [], 'lp_hits@2': [], 'lp_hits@5': [], 'lp_hits@10': []
                        }

        epoch_preds_labels = {"pos_pred": [], "neg_pred": [], "pos_labels": [], "neg_labels": []}
        
        assert model.training, "Training error: Model is not in training mode"
        step = 0
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            
            #remove the unique edge id from the edge features, as it's no longer needed
            batch.edge_attr = batch.edge_attr[:, 1:]

            negative_edge_sampling(batch, args)

            if args.edge_agg_type=='adamm':
                batch = ToMultigraph(batch)
            else:
                batch = create_hetero_obj(batch.x, batch.y, batch.edge_index, batch.edge_attr, batch.timestamps, args, batch.simp_edge_batch, batch)

            ''' Add port numberings after neighborhood sampling. ''' 
            if args.ports and args.ports_batch:
                # To be consistent, sample the edges for forward and backward edge types.
                assign_ports_with_cpp(batch) 
            
            optimizer.zero_grad()

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

            loss = lp_loss_fn(pos_pred, neg_pred)
            batch_metrics['loss'].append(loss.detach().item())

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * (pos_pred.numel() + neg_pred.numel())
            total_examples += (pos_pred.numel() + neg_pred.numel())

            # Free some GPU memory
            pos_pred, neg_pred, pos_labels, neg_labels = pos_pred.detach().cpu(), neg_pred.detach().cpu(), pos_labels.detach().cpu(), neg_labels.detach().cpu()
            loss = loss.detach().cpu()

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
                epoch_preds_labels[k].extend(v.tolist())

            step += 1
            global_step += 1
            # Log batch metrics
            if step % 200 == 0:
                logging.info(f'\nTrain ' + '| '.join([f'{k}: {v[-1]:.4g}' for k,v in batch_metrics.items()]))
                wandb.log({f'batch_{k}': v[-1] for k, v in batch_metrics.items()}, step=global_step)


       ## After epoch ends
        # Log epoch metrics
        epoch_metrics = {f'tr_{met}': np.mean(batch_metrics[met]) for met in ['loss', 'lp_mean_acc', 'lp_auc', 'lp_mrr', 'lp_hits@1', 'lp_hits@2', 'lp_hits@5', 'lp_hits@10']}
        epoch_preds_labels = {k: np.array(v) for k, v in epoch_preds_labels.items()}
        lp_metrics = lp_compute_metrics(**gnn_preds_labels)
        for k, v in lp_metrics.items():
            epoch_metrics['tr_' + k] = v
        logging.info("***** Train results - EPOCH {} *****".format(epoch))
        for k, v in epoch_metrics.items():
            logging.info(f" {k}: {v:.4g}")

        wandb.log(epoch_metrics, step=global_step)

        # Do evaluation
        # Clear CUDA cache before evaluating
        torch.cuda.empty_cache()
        
        val_metrics = evaluate_homo_lp(val_loader, val_inds, model, val_data, device, args, mode='eval')
        logging.info("***** {} - Eval results *****".format(epoch))
        for key in sorted(val_metrics.keys()):
            logging.info("  %s = %s" % (key, str(val_metrics[key])))

        te_metrics = evaluate_homo_lp(te_loader, te_inds, model, te_data, device, args, mode='test')
        logging.info("***** {} - Eval results *****".format(epoch))
        for key in sorted(te_metrics.keys()):
            logging.info("  %s = %s" % (key, str(te_metrics[key])))

        # Log eval metrics
        wandb.log(val_metrics, step=global_step)
        wandb.log(te_metrics, step=global_step)

    return model


def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1]) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1])
    
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

def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
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

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)
    
    #get the model
    sample_batch = next(iter(tr_loader))

    if isinstance(sample_batch, HeteroData):
        sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
        sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
    else:
        sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]

    if args.task == 'lp':
        negative_edge_sampling(sample_batch, args)
    
    if args.edge_agg_type=='adamm':
        if args.adamm_hetero:
            sample_batch = HeteroToMultigraph(sample_batch)
        else:
            sample_batch = ToMultigraph(sample_batch)
    else:
        if not isinstance(sample_batch, HeteroData):
            sample_batch = create_hetero_obj(sample_batch.x, sample_batch.y, sample_batch.edge_index, sample_batch.edge_attr, sample_batch.timestamps, args, sample_batch.simp_edge_batch, sample_batch)

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
    
    sample_batch.to(device)

    logging.info(summary(model, sample_batch))
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    if args.task == 'lp':
        if args.reverse_mp or args.adamm_hetero:
            model = train_hetero_lp(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
        else:
            model = train_homo_lp(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    else:
        if args.reverse_mp:
            model = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
        else:
            model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    
    wandb.finish()