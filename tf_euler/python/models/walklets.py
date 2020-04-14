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


class Walklets(base.UnsupervisedModel):
  """
  """

  def __init__(self, node_type, edge_type, max_id,
               dim, skip_len=[0], walk_len=3, walk_p=1, walk_q=1,
               left_win_size=1, right_win_size=1, num_negs=5,
               feature_idx=-1, feature_dim=0, use_id=True,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False, combiner='add',
               share_negs=False,
               *args, **kwargs):
    super(Walklets, self).__init__(
        node_type, edge_type, max_id, *args, **kwargs)
    self.node_type = node_type
    self.edge_type = edge_type
    self.max_id = max_id
    self.dim = dim
    self.walk_len = walk_len
    self.walk_p = walk_p
    self.walk_q = walk_q
    self.left_win_size = left_win_size
    self.right_win_size = right_win_size
    self.num_negs = num_negs
    self.share_negs = share_negs

    if not isinstance(skip_len, list):
      skip_len = [skip_len]
      
    skip_len = map(int, skip_len)
    self.skip_len = skip_len

    self.batch_size_ratio = 0
    
    max_skip_len = skip_len[0]
    for e in skip_len:
      if e > max_skip_len:
        max_skip_len = e
      
      if e < left_win_size:
        self.batch_size_ratio += walk_len - e - 1

      if e < right_win_size:
        self.batch_size_ratio += walk_len - e - 1

    self.left_win_size = min(self.left_win_size, max_skip_len+1)
    self.right_win_size = min(self.right_win_size, max_skip_len+1)
    self.max_distance = max(self.left_win_size, self.right_win_size)

    assert max_skip_len < self.max_distance
    assert self.batch_size_ratio > 0

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

  def to_sample(self, inputs):
    batch_size = tf.size(inputs)
    path = euler_ops.random_walk(
        inputs, [self.edge_type] * self.walk_len,
        p=self.walk_p,
        q=self.walk_q,
        default_node=-1)
    pair, distance = euler_ops.gen_pair(path, self.left_win_size, self.right_win_size)
    pair = tf.reshape(pair, [-1, 2])
    distance = tf.reshape(distance, [-1])
    distance = tf.cast(distance, dtype=tf.int32)
    pairs = tf.dynamic_partition(pair, distance, num_partitions=self.max_distance)
    pair_list = []
    for e in self.skip_len:
      if e < len(pairs):
        pair_list.append(pairs[e])
    pairs = tf.concat(pair_list, axis=0)

    src, pos = tf.split(pairs, [1, 1], axis=-1)
    src = tf.reshape(src, [-1, 1])
    pos = tf.reshape(pos, [-1, 1])
    negs = euler_ops.sample_node_with_src(tf.reshape(pos, [-1]), self.num_negs, self.share_negs)
    negs = tf.reshape(negs, [-1, self.num_negs])
    return src, pos, negs

  def target_encoder(self, inputs):
    return self._target_encoder(inputs)

  def context_encoder(self, inputs):
    return self._context_encoder(inputs)
