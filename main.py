from recbole.quick_start import run_recbole
import torch
import json
from models.model import SASRecFPlus

parameter_dict = {
    'use_gpu': True,
    'state': 'INFO',
    'gpu_id': 0,
    'show_progress': False,
    # model parameters
    'hidden_size': 64,
    'inner_size': 256,
    'n_layers': 2,
    'n_heads': 2,
    'layer_norm_eps': 1e-12,
    'initializer_range': 0.02,
    'hidden_act': 'gelu',
    'loss_type': 'CE',
    'pooling_mode': 'sum',
    'hidden_dropout_prob': 0.2,
    'attn_dropout_prob': 0.2,
    'enable_amp': True,
    'train_neg_sample_args': None,
    'MAX_ITEM_LIST_LENGTH': 200,
    'load_col': {
        'inter': ['user_id', 'item_id', 'timestamp'],
        'item': ['item_id', 'genre', 'release_year']
    },
    'selected_features': ['genre', 'release_year'],
    'embedding_size': 64,
    'epochs': 200,
    'train_batch_size': 128,
    'eval_batch_size': 128,
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

result = run_recbole(
    model= "SASRecF",
    dataset='ml-1m',
    config_dict=parameter_dict
)
