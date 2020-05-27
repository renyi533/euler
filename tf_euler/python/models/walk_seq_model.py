# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_euler.python import encoders
from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python.models import base
from tf_euler.python.utils.transformer_modules import ff, positional_encoding, multihead_attention

class WalkSeqModel(base.UnsupervisedModel):
  """
  """

  def __init__(self, node_type, edge_type, max_id,
               dim, walk_len=3, walk_p=1, walk_q=1,
               num_negs=5, cell='lstm', cell_layers=1,
               feature_idx=-1, feature_dim=0, use_id=True,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False, combiner='add',
               share_negs=False,
               *args, **kwargs):
    super(WalkSeqModel, self).__init__(
        node_type, edge_type, max_id, *args, **kwargs)
    self.node_type = node_type
    self.edge_type = edge_type
    self.max_id = max_id
    self.dim = dim
    self.walk_len = walk_len
    self.walk_p = walk_p
    self.walk_q = walk_q
    self.num_negs = num_negs
    self.share_negs = share_negs
    self.cell_layers = cell_layers
    self.batch_size_ratio = walk_len

    self._target_encoder = encoders.ShallowEncoder(
        dim=dim, feature_idx=feature_idx, feature_dim=feature_dim,
        max_id=max_id if use_id else -1,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id,
        embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
        combiner=combiner)
    self._context_encoder = encoders.ShallowEncoder(
        dim=dim, feature_idx=feature_idx, feature_dim=feature_dim,
        max_id=max_id if use_id else -1,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id,
        embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
        combiner=combiner)

    if cell == 'lstm':
      self.cells = [tf.compat.v1.nn.rnn_cell.BasicLSTMCell(dim) \
                    for _ in range(self.cell_layers)]
    elif cell == 'gru':
      self.cells = [tf.compat.v1.nn.rnn_cell.GRUCell(dim) \
                    for _ in range(self.cell_layers)]
    else:
      assert cell == 'transformer'
    
    if cell == 'transformer':
      self.stacked_cell = None
    else:
      self.stacked_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(self.cells, \
                    state_is_tuple=True)

  def to_seq_sample(self, inputs):
    batch_size = tf.size(inputs)
    path = euler_ops.random_walk(
        inputs, [self.edge_type] * self.walk_len,
        p=self.walk_p,
        q=self.walk_q,
        default_node=-1)

    src, _ = tf.split(path, [-1, 1], axis=1)
    _, pos = tf.split(path, [1, -1], axis=1)
    negs = euler_ops.sample_node_with_src(tf.reshape(pos, [-1]), self.num_negs, self.share_negs)
    negs = tf.reshape(negs, [batch_size, self.walk_len, self.num_negs])
    return src, pos, negs

  def _gen_seq_embedding(self, embedding, masks):
    if self.stacked_cell is not None: 
      initial_state = self.stacked_cell.zero_state(tf.shape(embedding)[0], dtype=tf.float32)
      rnn_embedding, last_state = tf.nn.dynamic_rnn(self.stacked_cell, embedding, 
                    initial_state=initial_state)
      return rnn_embedding
    else:
      embedding *= self.dim ** 0.5  # scale
      dec = embedding + positional_encoding(embedding, self.walk_len)

      for i in range(self.cell_layers):
        with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
          # Masked self-attention (Note that causality is True at this time)
          dec = multihead_attention(queries=dec,
                                    keys=dec,
                                    values=dec,
                                    key_masks=masks,
                                    num_heads=2,
                                    dropout_rate=0.0,
                                    training=True,
                                    causality=True,
                                    scope="self_attention")
          ### Feed Forward
          dec = ff(dec, num_units=[self.dim, self.dim])
      return dec

  def call(self, inputs):
    src, pos, negs = self.to_seq_sample(inputs)
    embedding = self.target_encoder(src)
    embedding_pos = self.context_encoder(pos)
    
    masks = tf.math.equal(src, -1)
    embedding = self._gen_seq_embedding(embedding, masks)
    bs = tf.shape(embedding)[0]
    embedding = tf.reshape(embedding, [bs * self.walk_len, 1, -1])
    embedding_pos = tf.reshape(embedding_pos, [bs * self.walk_len, 1, -1])

    negs_1d = tf.reshape(negs, [-1])
    uniq_negs, idx, counts = tf.unique_with_counts(negs_1d, 
                                                   out_idx=tf.int64)
    embedding_negs = self.context_encoder(uniq_negs)
    embedding_negs = tf.gather(embedding_negs, idx, axis=0)
    embedding_negs = tf.reshape(embedding_negs,
                                [bs * self.walk_len, self.num_negs,-1])

    loss, mrr = self.decoder(embedding, embedding_pos, embedding_negs)
    embedding = self.target_encoder(inputs)

    return base.ModelOutput(
        embedding=embedding, loss=loss, metric_name='mrr', metric=mrr)

  def target_encoder(self, inputs):
    return self._target_encoder(inputs)

  def context_encoder(self, inputs):
    return self._context_encoder(inputs)
