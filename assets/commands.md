- MEGA-GIN
```bash
python main.py --data Small_HI --model gin --emlps --reverse_mp --ego --flatten_edges --edge_agg_type gin --n_epochs 80 --save_model --task edge_class
```
- MEGA-PNA
```bash
python main.py --data Small_HI --model pna --emlps --reverse_mp --ego --flatten_edges --edge_agg_type pna --n_epochs 80 --save_model --task edge_class
```

- MEGA-GenAgg
```bash
python main.py --data Small_HI --model gin --emlps --reverse_mp --ego --flatten_edges --edge_agg_type gin --node_agg_type genagg --n_epochs 80 --save_model --task edge_class
```

| Different combinations of aggregation functions can be selected. For example;
- MEGA(GenAgg)-GIN
```bash
python main.py --data Small_HI --model gin --emlps --reverse_mp --ego --flatten_edges --edge_agg_type genagg --n_epochs 80 --save_model --task edge_class
```

## ETH Dataset
- MEGA-GIN
```bash
python main.py --data ETH-Kaggle --model gin --emlps --ego --reverse_mp --flatten_edges --edge_agg_type gin --task node_class --batch_size 4096 --n_epochs 80
```
- MEGA-PNA
```bash
python main.py --data ETH-Kaggle --model pna --emlps --ego --reverse_mp --flatten_edges --edge_agg_type pna --task node_class --batch_size 4096 --n_epochs 80
```
- MEGA-GenAgg
```bash
python main.py --data ETH-Kaggle --model gin --emlps --ego --reverse_mp --flatten_edges --edge_agg_type genagg --node_agg_type genagg --task node_class --batch_size 4096 --n_epochs 80
```

- For baseline ADAMM results
- ADAMM-GIN
```bash
python main.py --data ETH-Kaggle --model gin --emlps --ego --flatten_edges --edge_agg_type adamm --task node_class --batch_size 4096 --n_epochs 80
```
- ADAMM-PNA
```bash
python main.py --data ETH-Kaggle --model pna --emlps --ego --flatten_edges --edge_agg_type adamm --task node_class --batch_size 4096 --n_epochs 80
```
- ADAMM-GenAgg
```bash
python main.py --data ETH-Kaggle --model gin --node_agg_type genagg --emlps --ego --flatten_edges --edge_agg_type adamm --task node_class --batch_size 4096 --n_epochs 80
```