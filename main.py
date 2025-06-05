from recbole.quick_start import run_recbole
import torch
import json
from models.model import SASRecFPlus

parameter_dict = {
    'use_gpu': True,
    'state': 'INFO',
    'hidden_dropout_prob': 0.2,
    'attention_dropout_prob': 0.2,
    'gpu_id': 0,
    'enable_amp': True,
    'train_neg_sample_args': None,
    'seq_len': {'item_id_list': 200},
    'load_col': {
        'inter': ['user_id', 'item_id', 'timestamp'],
        # 'item': ['item_id', 'genre']
    },
    # 'selected_features': ['genre'],
    'embedding_size': 64,
    'epochs': 200,
    'train_batch_size': 4096,
    'eval_batch_size': 4096,
    'eval_args': {
        'group_by': 'user',
        'order': 'TO',
        'split': {'LS': 'valid_and_test'},
        'mode': 'uni100',
        'metrics': ['NDCG'],
        'topk': 10,
        'metric_decimal_place': 4,
        'valid_metric': 'NDCG@10'
    }
}

model = SASRecFPlus(config=parameter_dict, dataset='ml-1m')

result = run_recbole(
    model= SASRecFPlus(),
    dataset='ml-1m',
    config_dict=parameter_dict
)

with open('result.json', 'w') as f:
    json.dump(result, f, indent=4)
