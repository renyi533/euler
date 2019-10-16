from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_euler.python import encoders
from tf_euler.python.models import base
from base_model import SignedUnsupervisedModel


class SignedAsymGraphSage(SignedUnsupervisedModel):
  def __init__(self, node_type, edge_type, neg_edge_type, max_id, context_max_id,
               metapath, fanouts, dim, aggregator='mean', concat=False,
               feature_idx=-1, feature_dim=0, use_feature=None, use_id=False,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False, use_residual=False,
               *args, **kwargs):
    super(SignedAsymGraphSage, self).__init__(
        node_type, edge_type, neg_edge_type, max_id, context_max_id, *args, **kwargs)
    self._target_encoder = encoders.SageEncoder(
        metapath, fanouts, dim, aggregator, concat,
        feature_idx=feature_idx, feature_dim=feature_dim,
        max_id=max_id, use_id=use_id,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id,
        embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
        use_residual=use_residual)
    self._context_encoder =  encoders.ShallowEncoder(
          dim=dim, feature_idx=feature_idx, feature_dim=feature_dim,
          max_id=context_max_id if use_id else -1,
          sparse_feature_idx=sparse_feature_idx,
          sparse_feature_max_id=sparse_feature_max_id,
          embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
          combiner='add')


  def target_encoder(self, inputs):
    return self._target_encoder(inputs)

  def context_encoder(self, inputs):
    return self._context_encoder(inputs)

