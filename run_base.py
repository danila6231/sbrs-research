from recbole.quick_start import run_recbole
import torch
import json
from models.sas_rec_f_plus import SASRecFPlus

parameter_dict = {
    # ENVIRONMENT
    'use_gpu': True,
    'gpu_id': 0,
    'state': 'INFO',
    'reproducibility': True,
    'seed': 2020,
    'data_path': 'dataset/',
    'show_progress': False,
    # DATA
    'field_separator': '\t',
    'load_col': {
        'inter': ['transaction_date', 'customer_id', 'product_id'],
        'user': ['customer_id', 'age_group', 'pin_code']
    },
    'MAX_ITEM_LIST_LENGTH': 40,
    'USER_ID_FIELD': 'customer_id',
    'ITEM_ID_FIELD': 'product_id',
    'TIME_FIELD': 'transaction_date',
    # TRAINING
    'epochs': 200,
    'train_batch_size': 128,
    'train_neg_sample_args': None,
    'enable_amp': True,
    # EVALUATION

    # 'eval_args': {
    #     'group_by': 'user',
    #     'order': 'TO',
    #     'split': {'RS': [0.8, 0.1, 0.1]},
    #     'mode': 'full'
    # },
    'eval_args': {
        'group_by': 'user',
        'order': 'TO',
        'split': {'LS': 'valid_and_test'},
        'mode': 'uni100'
    },
    'metrics': ['NDCG', 'MRR', 'Hit'],
    'valid_metric': 'NDCG@10',
    'topk': 10,
    'metric_decimal_place': 4,
    'eval_batch_size': 128,
    # MODEL
    'embedding_size': 64,
    'hidden_size': 64,
    'feature_emb_hidden_size': 32,
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
}

result = run_recbole(
    model= "SASRec",
    dataset='ta-feng',
    config_dict=parameter_dict
)
