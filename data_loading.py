import pandas as pd
import numpy as np
import torch
import logging
import itertools
from data_util import GraphData, HeteroData, z_norm, create_hetero_obj, find_parallel_edges, assign_ports_with_cpp

def get_data(args, data_config):
    '''Loads the AML transaction data.
    
    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''

    transaction_file = f"{data_config['paths']['aml_data']}/{args.data}/formatted_transactions.csv" #replace this with your path to the respective AML data objects
    df_edges = pd.read_csv(transaction_file)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

    logging.info(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    logging.info(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    node_features = ['Feature']

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    simp_edge_batch = find_parallel_edges(edge_index) if args.flatten_edges else None

    n_days = int(timestamps.max() / (3600 * 24) + 1)
    n_samples = y.shape[0]
    logging.info(f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

    #data splitting
    daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], [] #irs = illicit ratios, inds = indices, trans = transactions
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_irs.append(y[day_inds].float().mean())
        weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])
    
    split_per = [0.6, 0.2, 0.2]
    daily_totals = np.array(daily_trans)
    d_ts = daily_totals
    I = list(range(len(d_ts)))
    split_scores = dict()
    for i,j in itertools.combinations(I, 2):
        if j >= i:
            split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v/split_totals_sum for v in split_totals]
            split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
            score = max(split_error) #- (split_totals_sum/total) + 1
            split_scores[(i,j)] = score
        else:
            continue

    i,j = min(split_scores, key=split_scores.get)
    #split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
    split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
    logging.info(f'Calculate split: {split}')

    #Now, we seperate the transactions based on their indices in the timestamp array
    split_inds = {k: [] for k in range(3)}
    for i in range(3):
        for day in split[i]:
            split_inds[i].append(daily_inds[day]) #split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately

    tr_inds = torch.cat(split_inds[0])
    val_inds = torch.cat(split_inds[1])
    te_inds = torch.cat(split_inds[2])

    logging.info(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
            f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0][:5]}")
    logging.info(f"Total val samples: {val_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
        f"{y[val_inds].float().mean() * 100:.2f}% || Val days: {split[1][:5]}")
    logging.info(f"Total test samples: {te_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
        f"{y[te_inds].float().mean() * 100:.2f}% || Test days: {split[2][:5]}")
    
    #Creating the final data objects
    tr_x, val_x, te_x = x, x, x
    e_tr = tr_inds.numpy()
    e_val = np.concatenate([tr_inds, val_inds])

    tr_edge_index,  tr_edge_attr,  tr_y,  tr_edge_times  = edge_index[:,e_tr],  edge_attr[e_tr],  y[e_tr],  timestamps[e_tr]
    val_edge_index, val_edge_attr, val_y, val_edge_times = edge_index[:,e_val], edge_attr[e_val], y[e_val], timestamps[e_val]
    te_edge_index,  te_edge_attr,  te_y,  te_edge_times  = edge_index,          edge_attr,        y,        timestamps

    if args.flatten_edges:
        tr_simp_edge_batch, val_simp_edge_batch, te_simp_edge_batch = simp_edge_batch[e_tr], simp_edge_batch[e_val], simp_edge_batch
        tr_data = GraphData (x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr,  timestamps=tr_edge_times , simp_edge_batch = tr_simp_edge_batch)
        val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times, simp_edge_batch = val_simp_edge_batch)
        te_data = GraphData (x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr,  timestamps=te_edge_times , simp_edge_batch = te_simp_edge_batch)
    else:
        tr_simp_edge_batch, val_simp_edge_batch, te_simp_edge_batch = None, None, None
        tr_data = GraphData (x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr,  timestamps=tr_edge_times )
        val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times)
        te_data = GraphData (x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr,  timestamps=te_edge_times )

    #Adding ports and time-deltas if applicable
    if args.ports and not args.ports_batch:
        logging.info(f"Start: adding ports")
        assign_ports_with_cpp(tr_data, process_batch=False)
        assign_ports_with_cpp(val_data, process_batch=False)
        assign_ports_with_cpp(te_data, process_batch=False)
        logging.info(f"Done: adding ports")
    if args.tds:
        logging.info(f"Start: adding time-deltas")
        tr_data.add_time_deltas()
        val_data.add_time_deltas()
        te_data.add_time_deltas()
        logging.info(f"Done: adding time-deltas")
    
    #Normalize data
    tr_data.x = val_data.x = te_data.x = z_norm(tr_data.x)
    if not args.model == 'rgcn':
        tr_data.edge_attr, val_data.edge_attr, te_data.edge_attr = z_norm(tr_data.edge_attr), z_norm(val_data.edge_attr), z_norm(te_data.edge_attr)
    else:
        tr_data.edge_attr[:, :-1], val_data.edge_attr[:, :-1], te_data.edge_attr[:, :-1] = z_norm(tr_data.edge_attr[:, :-1]), z_norm(val_data.edge_attr[:, :-1]), z_norm(te_data.edge_attr[:, :-1])

    #Create heterogenous if reverese MP is enabled
    #TODO: if I observe wierd behaviour, maybe add .detach.clone() to all torch tensors, but I don't think they're attached to any computation graph just yet
    if args.reverse_mp  or args.adamm_hetero:
        tr_data = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr, tr_data.timestamps, args, tr_simp_edge_batch)
        val_data = create_hetero_obj(val_data.x,  val_data.y,  val_data.edge_index,  val_data.edge_attr, val_data.timestamps, args, val_simp_edge_batch)
        te_data = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr, te_data.timestamps, args, te_simp_edge_batch)
    
    logging.info(f'train data object: {tr_data}')
    logging.info(f'validation data object: {val_data}')
    logging.info(f'test data object: {te_data}')

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds


def get_eth_data(args, data_config):
    '''Loads the ETH Phishing data.
    
    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''

    nodes = pd.read_csv(f"{data_config['paths']['aml_data']}/{args.data}/nodes.csv")
    edges = pd.read_csv(f"{data_config['paths']['aml_data']}/{args.data}/edges.csv")

    logging.info(f'Available Edge Features: {edges.columns.tolist()}')
    logging.info(f"Number of Nodes: {nodes.shape[0]}")
    logging.info(f"Number of Edges: {edges.shape[0]}")
    logging.info(f"Number of Phishing Nodes: {nodes['label'].sum()}")
    logging.info(f"Illicit Ratio: {nodes['label'].sum()/ nodes.shape[0] * 100 :.2f}%")

    nodes = nodes.sort_values(by='first_transaction').reset_index(drop=True)

    assign_dict = {}
    for row in nodes.itertuples():
        assign_dict[row[1]] = row[0]

    def assign_node_ids(node_id):
        return assign_dict[node_id] 

    edges['to_address'] = edges['to_address'].apply(assign_node_ids)
    edges['from_address'] = edges['from_address'].apply(assign_node_ids)
    nodes.drop(columns=['node'], inplace=True)

    edges['block_timestamp'] = edges['block_timestamp'] - edges['block_timestamp'].min()

    edge_features = ['nonce', 'value', 'gas', 'gas_price', 'block_timestamp']
    node_features = ['Feature']

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    max_n_id = nodes.shape[0]

    splits = [0.65, 0.15, 0.20]

    t1 = nodes.iloc[int(max_n_id * splits[0])]['first_transaction']
    t2 = nodes.iloc[int(max_n_id * (splits[0] + splits[1]))]['first_transaction']

    tr_nodes = nodes.loc[nodes['first_transaction'] <= t1]
    val_nodes = nodes.loc[nodes['first_transaction'] <= t2]
    te_nodes = nodes

    tr_nodes_max_id = tr_nodes.index[-1]
    val_nodes_max_id = val_nodes.index[-1]
    te_nodes_max_id = te_nodes.index[-1]

    tr_inds = torch.arange(0, tr_nodes_max_id+1)
    val_inds = torch.arange(tr_nodes_max_id+1, val_nodes_max_id+1)
    te_inds = torch.arange(val_nodes_max_id+1, te_nodes_max_id+1)


    logging.info(f"Total train samples: {tr_nodes.shape[0] / nodes.shape[0] * 100 :.2f}% || IR: "
            f"{tr_nodes['label'].mean() * 100 :.2f}%")
    logging.info(f"Total validation samples: {val_inds.shape[0] / nodes.shape[0] * 100 :.2f}% || IR: "
            f"{val_nodes.loc[val_inds,'label'].mean() * 100 :.2f}%")
    logging.info(f"Total test samples: {te_inds.shape[0] / nodes.shape[0] * 100 :.2f}% || IR: "
            f"{te_nodes.loc[te_inds, 'label'].mean() * 100 :.2f}%")
    

    tr_nodes_max_id = tr_nodes.index[-1]
    val_nodes_max_id = val_nodes.index[-1]
    te_nodes_max_id = te_nodes.index[-1]

    split_name = []
    for row in edges.itertuples():
        if row[2] <= tr_nodes_max_id and row[3] <= tr_nodes_max_id:
            split_name.append('train')
            continue
        elif row[2] <= val_nodes_max_id and row[3] <= val_nodes_max_id:
            split_name.append('val')
        else:
            split_name.append('test')


    edges['split'] = split_name


    tr_edges = edges.loc[edges['split'] == 'train']
    val_edges = edges.loc[(edges['split'] == 'train') | (edges['split'] == 'val')]
    te_edges = edges 

    tr_x = torch.tensor(np.ones(tr_nodes.shape[0])).float()
    tr_edge_index = torch.LongTensor(tr_edges.loc[:, ['from_address', 'to_address']].to_numpy().T)
    tr_edge_attr = torch.tensor(tr_edges.loc[:, edge_features].to_numpy()).float()
    tr_edge_times = torch.Tensor(tr_edges['block_timestamp'].to_numpy())
    tr_y = torch.LongTensor(tr_nodes['label'].to_numpy())
    tr_simp_edge_batch = find_parallel_edges(tr_edge_index)

    val_x = torch.tensor(np.ones(val_nodes.shape[0])).float()
    val_edge_index = torch.LongTensor(val_edges.loc[:, ['from_address', 'to_address']].to_numpy().T)
    val_edge_attr = torch.tensor(val_edges.loc[:, edge_features].to_numpy()).float()
    val_edge_times = torch.Tensor(val_edges['block_timestamp'].to_numpy())
    val_y = torch.LongTensor(val_nodes['label'].to_numpy())
    val_simp_edge_batch = find_parallel_edges(val_edge_index)

    te_x = torch.tensor(np.ones(te_nodes.shape[0])).float()
    te_edge_index = torch.LongTensor(te_edges.loc[:, ['from_address', 'to_address']].to_numpy().T)
    te_edge_attr = torch.tensor(te_edges.loc[:, edge_features].to_numpy()).float()
    te_edge_times = torch.Tensor(te_edges['block_timestamp'].to_numpy())
    te_y = torch.LongTensor(te_nodes['label'].to_numpy())
    te_simp_edge_batch = find_parallel_edges(te_edge_index)

    if args.flatten_edges:
        tr_data = GraphData (x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr, timestamps=tr_edge_times , simp_edge_batch = tr_simp_edge_batch)
        val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr,timestamps=val_edge_times, simp_edge_batch = val_simp_edge_batch)
        te_data = GraphData (x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr, timestamps=te_edge_times , simp_edge_batch = te_simp_edge_batch)
    else:
        tr_data = GraphData (x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr , timestamps=tr_edge_times )
        val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times)
        te_data = GraphData (x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr , timestamps=te_edge_times )


    if args.ports and not args.ports_batch:
        logging.info(f"Start: adding ports")
        assign_ports_with_cpp(tr_data, process_batch=False)
        assign_ports_with_cpp(val_data, process_batch=False)
        assign_ports_with_cpp(te_data, process_batch=False)
        logging.info(f"Done: adding ports")


    tr_data.x, val_data.x, te_data.x = z_norm(tr_data.x), z_norm(val_data.x), z_norm(te_data.x)

    tr_data.edge_attr, val_data.edge_attr, te_data.edge_attr = z_norm(tr_data.edge_attr), z_norm(val_data.edge_attr), z_norm(te_data.edge_attr)

    if args.reverse_mp:
        tr_data = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr, tr_data.timestamps, args, tr_simp_edge_batch)
        val_data = create_hetero_obj(val_data.x,  val_data.y,  val_data.edge_index,  val_data.edge_attr, val_data.timestamps, args, val_simp_edge_batch)
        te_data = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr, te_data.timestamps, args, te_simp_edge_batch)

    logging.info(f'train data object: {tr_data}')
    logging.info(f'validation data object: {val_data}')
    logging.info(f'test data object: {te_data}')

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds



def get_eth_kaggle_data(args, data_config):
    '''Loads the ETH Phishing data.
    
    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''

    nodes = pd.read_csv(f"{data_config['paths']['aml_data']}/{args.data}/nodes-kaggle.csv")
    edges = pd.read_csv(f"{data_config['paths']['aml_data']}/{args.data}/edges-kaggle.csv")
    nodes.drop('Unnamed: 0', axis=1, inplace=True)
    edges.drop('Unnamed: 0', axis=1, inplace=True)

    logging.info(f'Available Edge Features: {edges.columns.tolist()}')
    logging.info(f"Number of Nodes: {nodes.shape[0]}")
    logging.info(f"Number of Edges: {edges.shape[0]}")
    logging.info(f"Number of Phishing Nodes: {nodes['label'].sum()}")
    logging.info(f"Illicit Ratio: {nodes['label'].sum()/ nodes.shape[0] * 100 :.2f}%")

    nodes = nodes.sort_values(by='first_transaction').reset_index(drop=True)

    assign_dict = {}
    for row in nodes.itertuples():
        assign_dict[row[1]] = row[0]

    def assign_node_ids(node_id):
        return assign_dict[node_id] 

    edges['to_address'] = edges['to_address'].apply(assign_node_ids)
    edges['from_address'] = edges['from_address'].apply(assign_node_ids)
    nodes.drop(columns=['node'], inplace=True)

    edge_features = ['amount', 'timestamp']
    node_features = ['Feature']

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    max_n_id = nodes.shape[0]

    splits = [0.65, 0.15, 0.20]

    t1 = nodes.iloc[int(max_n_id * splits[0])]['first_transaction']
    t2 = nodes.iloc[int(max_n_id * (splits[0] + splits[1]))]['first_transaction']

    tr_nodes = nodes.loc[nodes['first_transaction'] <= t1]
    val_nodes = nodes.loc[nodes['first_transaction'] <= t2]
    te_nodes = nodes

    tr_nodes_max_id = tr_nodes.index[-1]
    val_nodes_max_id = val_nodes.index[-1]
    te_nodes_max_id = te_nodes.index[-1]

    tr_inds = torch.arange(0, tr_nodes_max_id+1)
    val_inds = torch.arange(tr_nodes_max_id+1, val_nodes_max_id+1)
    te_inds = torch.arange(val_nodes_max_id+1, te_nodes_max_id+1)


    logging.info(f"Total train samples: {tr_nodes.shape[0] / nodes.shape[0] * 100 :.2f}% || IR: "
            f"{tr_nodes['label'].mean() * 100 :.2f}%")
    logging.info(f"Total validation samples: {val_inds.shape[0] / nodes.shape[0] * 100 :.2f}% || IR: "
            f"{val_nodes.loc[val_inds.numpy(),'label'].mean() * 100 :.2f}%")
    logging.info(f"Total test samples: {te_inds.shape[0] / nodes.shape[0] * 100 :.2f}% || IR: "
            f"{te_nodes.loc[te_inds.numpy(), 'label'].mean() * 100 :.2f}%")
    

    tr_nodes_max_id = tr_nodes.index[-1]
    val_nodes_max_id = val_nodes.index[-1]
    te_nodes_max_id = te_nodes.index[-1]

    split_name = []
    for row in edges.itertuples():
        '''
        row[0]: index
        row[1]: from_address
        row[2]: to_address
        row[3]: amount
        row[4]: timestamp
        '''
        if row[1] <= tr_nodes_max_id and row[2] <= tr_nodes_max_id:
            if row[4] <= t1:
                split_name.append('train')
            elif row[4] > t1 and row[4] <= t2:
                split_name.append('val')
            else:
                split_name.append('test') 
            continue 
        elif row[1] <= val_nodes_max_id and row[2] <= val_nodes_max_id:
            if row[4] <= t2:
                split_name.append('val')
            else:
                split_name.append('test') 
            continue
        else:
            split_name.append('test')


    edges['split'] = split_name
    edges['timestamp'] = edges['timestamp'] - edges['timestamp'].min()


    tr_edges = edges.loc[edges['split'] == 'train']
    val_edges = edges.loc[(edges['split'] == 'train') | (edges['split'] == 'val')]
    te_edges = edges 

    tr_x = torch.tensor(np.ones(tr_nodes.shape[0])).float()
    tr_edge_index = torch.LongTensor(tr_edges.loc[:, ['from_address', 'to_address']].to_numpy().T)
    tr_edge_attr = torch.tensor(tr_edges.loc[:, edge_features].to_numpy()).float()
    tr_edge_times = torch.Tensor(tr_edges['timestamp'].to_numpy())
    tr_y = torch.LongTensor(tr_nodes['label'].to_numpy())
    tr_simp_edge_batch = find_parallel_edges(tr_edge_index)

    val_x = torch.tensor(np.ones(val_nodes.shape[0])).float()
    val_edge_index = torch.LongTensor(val_edges.loc[:, ['from_address', 'to_address']].to_numpy().T)
    val_edge_attr = torch.tensor(val_edges.loc[:, edge_features].to_numpy()).float()
    val_edge_times = torch.Tensor(val_edges['timestamp'].to_numpy())
    val_y = torch.LongTensor(val_nodes['label'].to_numpy())
    val_simp_edge_batch = find_parallel_edges(val_edge_index)

    te_x = torch.tensor(np.ones(te_nodes.shape[0])).float()
    te_edge_index = torch.LongTensor(te_edges.loc[:, ['from_address', 'to_address']].to_numpy().T)
    te_edge_attr = torch.tensor(te_edges.loc[:, edge_features].to_numpy()).float()
    te_edge_times = torch.Tensor(te_edges['timestamp'].to_numpy())
    te_y = torch.LongTensor(te_nodes['label'].to_numpy())
    te_simp_edge_batch = find_parallel_edges(te_edge_index)

    if args.flatten_edges:
        tr_data = GraphData (x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr, timestamps=tr_edge_times , simp_edge_batch = tr_simp_edge_batch)
        val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr,timestamps=val_edge_times, simp_edge_batch = val_simp_edge_batch)
        te_data = GraphData (x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr, timestamps=te_edge_times , simp_edge_batch = te_simp_edge_batch)
    else:
        tr_data = GraphData (x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr , timestamps=tr_edge_times )
        val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times)
        te_data = GraphData (x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr , timestamps=te_edge_times )


    if args.ports and not args.ports_batch:
        logging.info(f"Start: adding ports")
        assign_ports_with_cpp(tr_data, process_batch=False)
        assign_ports_with_cpp(val_data, process_batch=False)
        assign_ports_with_cpp(te_data, process_batch=False)
        # tr_data.add_ports()
        # val_data.add_ports()
        # te_data.add_ports()
        logging.info(f"Done: adding ports")


    tr_data.x, val_data.x, te_data.x = z_norm(tr_data.x), z_norm(val_data.x), z_norm(te_data.x)

    tr_data.edge_attr, val_data.edge_attr, te_data.edge_attr = z_norm(tr_data.edge_attr), z_norm(val_data.edge_attr), z_norm(te_data.edge_attr)

    if args.reverse_mp:
        tr_data = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr, tr_data.timestamps, args, tr_simp_edge_batch)
        val_data = create_hetero_obj(val_data.x,  val_data.y,  val_data.edge_index,  val_data.edge_attr, val_data.timestamps, args, val_simp_edge_batch)
        te_data = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr, te_data.timestamps, args, te_simp_edge_batch)

    logging.info(f'train data object: {tr_data}')
    logging.info(f'validation data object: {val_data}')
    logging.info(f'test data object: {te_data}')

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds