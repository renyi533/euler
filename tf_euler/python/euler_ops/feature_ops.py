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

from tf_euler.python.euler_ops import base

def _get_thread_num(data, thread_num):
  if thread_num <= 0:
    thread_num = tf.maximum(tf.shape(data)[0] // 3000, 1)
    thread_num = tf.minimum(thread_num, 5)
  return thread_num

def _iter_body(i, state):
  ta, size = state
  out_ta = ta.write(i, size)
  return i+1, (out_ta, size)

def _split_input_data(data_list, thread_num):
  size = tf.shape(data_list)[0]
  split_size = tf.maximum(size // thread_num, 1)

  split_size = tf.reshape(split_size, [1,1])
  ta = tf.TensorArray(dtype=tf.int32, size=thread_num, infer_shape=False)
  init_state = (0, (ta, split_size))
  condition = lambda i, _: i < thread_num-1
  _, (ta_final, _) = tf.while_loop(condition, _iter_body, init_state)
  ta_final = tf.cond(tf.equal(thread_num, 1), lambda: ta_final.write(0, split_size), 
                     lambda: ta_final.write(thread_num-1, tf.reshape(-1,[1,1])) )
  split_dims = tf.reshape(ta_final.concat(), [thread_num])
  
  split_data_list = tf.split(data_list, split_dims, axis=0)
  return split_data_list

def _get_sparse_feature(nodes_or_edges, feature_ids, op, thread_num,
    default_values=None):
  if default_values is None:
    default_values = [0] * len(feature_ids)

  split_data_list = _split_input_data(nodes_or_edges, thread_num)
  split_result_list = [op(split_data, feature_ids, default_values,
      len(feature_ids)) for split_data in split_data_list]
  split_sp = []
  for i in range(len(split_result_list)):
    split_sp.append(
        [tf.SparseTensor(*sp) for sp in zip(*split_result_list[i])])
  split_sp_transpose = map(list, zip(*split_sp))
  return [tf.sparse_concat(axis=0, sp_inputs=sp,
      expand_nonconcat_dim=True) for sp in split_sp_transpose]

def get_sparse_feature(nodes, feature_ids, thread_num=1, default_values=None):
  """
  Fetch sparse features of nodes.

  Args:
    nodes: A 1-d `Tensor` of `int64`.
    feature_ids: A list of `int`. Specify uint64 feature ids in graph to fetch
      features for nodes.
    default_values: A `int`. Specify value to fill when there is no specific
      features for specific nodes.

  Return:
    A list of `SparseTensor` with the same length as `feature_ids`.
  """
  uniq_nodes, orig_idx = tf.unique(nodes, out_idx=tf.dtypes.int64)
  thread_num = _get_thread_num(nodes, thread_num)
  uniq_sp_features = _get_sparse_feature(uniq_nodes, feature_ids,
                             base._LIB_OP.get_sparse_feature, thread_num, default_values)
  sp_features = [base._LIB_OP.sparse_gather(orig_idx, t.indices, t.values, t.dense_shape) \
                  for t in uniq_sp_features]
  return [tf.SparseTensor(*sp) for sp in sp_features]

def get_edge_sparse_feature(edges, feature_ids, thread_num=1, default_values=None):
  """
  Args:
    edges: A 2-D `Tensor` of `int64`, with shape `[num_edges, 3]`.
    feature_ids: A list of `int`. Specify uint64 feature ids in graph to fetch
      features for edges.
    default_values: A `int`. Specify value to fill when there is no specific
      features for specific edges.

  Return:
    A list of `SparseTensor` with the same length as `feature_ids`.
  """
  thread_num = _get_thread_num(edges, thread_num)
  return _get_sparse_feature(edges, feature_ids,
                             base._LIB_OP.get_edge_sparse_feature, thread_num, default_values)


def _get_dense_feature(nodes_or_edges, feature_ids, dimensions, op, thread_num):
  split_data_list = _split_input_data(nodes_or_edges, thread_num)
  split_result_list = [op(split_data, feature_ids, dimensions, N=len(feature_ids))
      for split_data in split_data_list]
  split_result_list_transpose = map(list, zip(*split_result_list))
  return [tf.concat(split_dense, 0)
      for split_dense in split_result_list_transpose]

def get_dense_feature(nodes, feature_ids, dimensions, thread_num=1):
  """
  Fetch dense features of nodes.

  Args:
    nodes: A 1-d `Tensor` of `int64`.
    feature_ids: A list of `int`. Specify float feature ids in graph to fetch
      features for nodes.
    dimensions: A list of `int`. Specify dimensions of each feature.

  Return:
    A list of `Tensor` with the same length as `feature_ids`.
  """
  uniq_nodes, orig_idx = tf.unique(nodes, out_idx=tf.dtypes.int64)

  thread_num = _get_thread_num(nodes, thread_num)

  uniq_features = _get_dense_feature(uniq_nodes, feature_ids, dimensions,
            base._LIB_OP.get_dense_feature, thread_num)
  return [tf.gather(t, orig_idx, axis=0) for t in uniq_features]

def get_edge_dense_feature(edges, feature_ids, dimensions, thread_num=1):
  """
  Fetch dense features of edges.

  Args:
    nodes: A 2-d `Tensor` of `int64`, with shape `[num_edges, 3]`.
    feature_ids: A list of `int`. Specify float feature ids in graph to fetch
      features for edges.
    dimensions: A list of `int`. Specify dimensions of each feature.

  Return:
    A list of `Tensor` with the same length as `feature_ids`.
  """
  thread_num = _get_thread_num(edges, thread_num)
  return _get_dense_feature(edges, feature_ids, dimensions,
            base._LIB_OP.get_edge_dense_feature, thread_num)

def _get_binary_feature(nodes_or_edges, feature_ids, op, thread_num):
  split_data_list = _split_input_data(nodes_or_edges, thread_num)
  split_result_list = [op(split_data, feature_ids, N=len(feature_ids))
      for split_data in split_data_list]
  split_result_list_transpose = map(list, zip(*split_result_list))
  return [tf.concat(split_binary, 0)
      for split_binary in split_result_list_transpose]

def get_binary_feature(nodes, feature_ids, thread_num=1):
  """
  Fetch binary features of nodes.

  Args:
    nodes: A 1-d `Tensor` of `int64`.
    feature_ids: A list of `int`. Specify uint64 feature ids in graph to fetch
      features for nodes.

  Return:
    A list of `String Tensor` with the same length as `feature_ids`.
  """
  thread_num = _get_thread_num(nodes, thread_num)
  return _get_binary_feature(nodes, feature_ids,
                             base._LIB_OP.get_binary_feature, thread_num)


def get_edge_binary_feature(edges, feature_ids, thread_num=1):
  """
  Fetch binary features of edges.

  Args:
    edges: A 2-d `Tensor` of `int64`, with shape `[num_edges, 3]`.
    feature_ids: A list of `int`. Specify uint64 feature ids in graph to fetch
      features for nodes.

  Return:
    A list of `String Tensor` with the same length as `feature_ids`.
  """
  thread_num = _get_thread_num(edges, thread_num)
  return _get_binary_feature(edges, feature_ids,
                             base._LIB_OP.get_edge_binary_feature, thread_num)
