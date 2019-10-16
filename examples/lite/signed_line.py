from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_euler.python import encoders
from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python.models import base
from base_model import SignedUnsupervisedModel

class SignedLINE(SignedUnsupervisedModel):
  """
  Implementation of LINE model.
  """

  def __init__(self, node_type, edge_type, neg_edge_type, max_id, context_max_id, dim, order=1,
               feature_idx=-1, feature_dim=0, use_id=True,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False, combiner='add',
               *args, **kwargs):
    super(SignedLINE, self).__init__(node_type, edge_type, neg_edge_type, max_id, context_max_id, *args, **kwargs)

    if order == 1:
      order = 'first'
    if order == 2:
      order = 'second'

    self._target_encoder = encoders.ShallowEncoder(
        dim=dim, feature_idx=feature_idx, feature_dim=feature_dim,
        max_id=max_id if use_id else -1,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id,
        embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
        combiner=combiner)
    if order == 'first':
      self._context_encoder = self._target_encoder
    elif order == 'second':
      self._context_encoder = encoders.ShallowEncoder(
          dim=dim, feature_idx=feature_idx, feature_dim=feature_dim,
          max_id=context_max_id if use_id else -1,
          sparse_feature_idx=sparse_feature_idx,
          sparse_feature_max_id=sparse_feature_max_id,
          embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
          combiner=combiner)
    else:
      raise ValueError('LINE order must be one of 1, 2, "first", or "second"'
                       'got {}:'.format(order))

  def target_encoder(self, inputs):
    return self._target_encoder(inputs)

  def context_encoder(self, inputs):
    return self._context_encoder(inputs)

