# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:32
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
SASRecFPlus
################################################
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer, ContextSeqEmbAbstractLayer

class UltimateFeatureSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For feature-rich sequential recommenders, return item features embedding matrices according to
    selected features."""

    def __init__(
        self, dataset, embedding_size, selected_item_features, selected_user_features, pooling_mode, device
    ):
        super(UltimateFeatureSeqEmbLayer, self).__init__()

        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = self.dataset.get_user_feature().to(self.device)
        self.item_feat = self.dataset.get_item_feature().to(self.device)

        self.field_names = {
            "user": selected_user_features,
            "item": selected_item_features
        }

        self.types = ["user", "item"]
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ["mean", "max", "sum"]
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()

class SASRecFPlus(SequentialRecommender):
    """This is an extension of SASRec, which concatenates item representations and item attribute representations
    as the input to the model.
    """

    def __init__(self, config, dataset):
        super(SASRecFPlus, self).__init__(config, dataset)

        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.selected_item_features = config['selected_item_features']
        self.selected_user_features = config['selected_user_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.num_item_feature_field = len(config['selected_item_features'])
        self.num_user_feature_field = len(config['selected_user_features'])

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.user_embedding = nn.Embedding(self.n_users, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
        
        # Only create feature embedding layer if features are selected
        if self.num_item_feature_field > 0 or self.num_user_feature_field > 0:
            self.feature_embed_layer = UltimateFeatureSeqEmbLayer(
                dataset,
                self.hidden_size,
                self.selected_item_features,
                self.selected_user_features,
                self.pooling_mode,
                self.device
            )
            # Now concat layer input size is 2*hidden_size (original + averaged features)
            self.item_concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.user_concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.trm_encoder = TransformerEncoder(n_layers=self.n_layers, n_heads=self.n_heads,
                                              hidden_size=self.hidden_size, inner_size=self.inner_size,
                                              hidden_dropout_prob=self.hidden_dropout_prob,
                                              attn_dropout_prob=self.attn_dropout_prob,
                                              hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def _concat_features(self, emb, sparse_embedding, dense_embedding):
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)
        
        # If no features are present, return the original embedding
        if len(feature_table) == 0:
            return emb
            
        # Concatenate all features along the last dimension
        feature_table = torch.cat(feature_table, dim=-2)  # [B, L, num_features, H]
        
        # Average across the features dimension to maintain hidden size
        feature_emb = torch.mean(feature_table, dim=-2)  # [B, L, H]
        
        # Concatenate with original embedding
        input_concat = torch.cat((emb, feature_emb), -1)  # [B, L, 2*H]
        return input_concat

    def forward(self, user, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        user_emb = self.user_embedding(user).unsqueeze(1)

        # position embedding
        position_ids = torch.arange(item_seq.size(1) + 1, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand(item_seq.size(0), item_seq.size(1) + 1)
        position_embedding = self.position_embedding(position_ids)

        if len(self.selected_item_features) > 0 or len(self.selected_user_features) > 0:
            sparse_embedding, dense_embedding = self.feature_embed_layer(user.unsqueeze(1), item_seq)
            item_sparse_embedding = sparse_embedding['item']
            item_dense_embedding = dense_embedding['item']
            user_sparse_embedding = sparse_embedding['user']
            user_dense_embedding = dense_embedding['user']
            
            item_input_concat = self._concat_features(item_emb, item_sparse_embedding, item_dense_embedding)
            user_input_concat = self._concat_features(user_emb, user_sparse_embedding, user_dense_embedding)
            
            print("item_input_concat", item_input_concat.shape)
            print("user_input_concat", user_input_concat.shape)
            item_input_emb = self.item_concat_layer(item_input_concat)
            user_input_emb = self.user_concat_layer(user_input_concat)
        else:
            # If no features are selected, use the original embeddings
            item_input_emb = item_emb
            user_input_emb = user_emb

        input_emb = torch.cat([user_input_emb, item_input_emb], dim=1)
        
        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        combined_seq = torch.cat([user.unsqueeze(1), item_seq], dim=1)
        extended_attention_mask = self.get_attention_mask(combined_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask,
                                      output_all_encoded_layers=True)
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output # [B H]


    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(user, item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss


    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(user, item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores


    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        seq_output = self.forward(user, item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores
