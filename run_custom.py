from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from models.model import SASRecFPlus
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

parameter_dict = {
    'use_gpu': True,
    'state': 'INFO',
    'gpu_id': 0,
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
    'seq_len': {'item_id': 200},
    'load_col': {
        'inter': ['user_id', 'item_id', 'timestamp'],
        'item': ['item_id', 'genre', 'release_year'],
        'user': ['user_id', 'gender', 'age', 'occupation']
    },
    'selected_features': ['genre', 'release_year'],
    'user_selected_features': ['gender', 'age', 'occupation'],
    'embedding_size': 64,
    'epochs': 200,
    'train_batch_size': 128,
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

if __name__ == '__main__':

    config = Config(model=SASRecFPlus, dataset='ml-1m', config_dict=parameter_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = SASRecFPlus(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))