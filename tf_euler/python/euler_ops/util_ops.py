# Copyright 2018 Alibaba Inc. All Rights Conserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import os

import tensorflow as tf

from tf_euler.python.euler_ops import base
from tensorflow.python.ops import variables

inflate_idx = base._LIB_OP.inflate_idx
sparse_gather = base._LIB_OP.sparse_gather
euler_hash_fid = base._LIB_OP.euler_hash_fid
euler_hash_fid_v2 = base._LIB_OP.hash_to_fid
reciprocal_rank_weight = base._LIB_OP.reciprocal_rank_weight

def hash_fid_v2(fids, hash_space, partition=None, erase=False):
  with tf.variable_scope('hash_fids', 
                         reuse=tf.AUTO_REUSE):
    if partition is None:
      param = tf.get_variable('hash_param',
                              [hash_space,1],
                              dtype=tf.int64,
                              trainable=False,
                              initializer=tf.constant_initializer([0]))
    else:
      param = tf.get_variable('hash_param',
                              [hash_space,1],
                              dtype=tf.int64,
                              trainable=False,
                              partitioner=tf.fixed_size_partitioner(partition, axis=0),
                              initializer=tf.constant_initializer([0]))


  params = param
  if isinstance(param, variables.PartitionedVariable):
    params = list(param)  # Iterate to get the underlying Variables.
  elif not isinstance(params, list):
    params = [params]

  orig_fids = fids
  fids, idx = tf.unique(fids, out_idx=tf.dtypes.int32)
  ret = None
  np = len(params)
  if np == 1:
    with tf.colocate_with(params[0]):
      ret = euler_hash_fid_v2(params[0], fids, 0, erase=erase)
  else:
    flat_ids = tf.reshape(fids, [-1])
    original_indices = tf.range(tf.size(flat_ids))
    p_assignments = tf.abs(flat_ids) % np
    new_ids = flat_ids // np

    p_assignments = tf.cast(p_assignments, tf.dtypes.int32)
    gather_ids = tf.dynamic_partition(new_ids, p_assignments, np)
    pindices = tf.dynamic_partition(original_indices,
                                    p_assignments, np)
    partitioned_result = []
    start = 0
    for p in xrange(np):
      pids = gather_ids[p]
      with tf.colocate_with(params[p]):
        result = euler_hash_fid_v2(params[p], pids, start, erase=erase)
        start = start + tf.shape(params[p], out_type=tf.dtypes.int64)[0]
      partitioned_result.append(result)
    ret = tf.dynamic_stitch(pindices, 
                            partitioned_result)
  ret = tf.reshape(ret, [-1])
  return tf.gather(ret, idx, axis=0)

def hash_fid_v3(v, fids, erase=False):
  params = []
  param_size = []
  with tf.variable_scope('hash_fids',
                         reuse=tf.AUTO_REUSE,
                         partitioner=tf.fixed_size_partitioner(1, axis=0)):
    var = v
    if isinstance(v, variables.PartitionedVariable):
      var = list(v)  # Iterate to get the underlying Variables.
    elif not isinstance(var, list):
      var = [var]

    for i in range(len(var)):
      with tf.colocate_with(var[i]):
        param_size.append(var[i].get_shape()[0])
        param = tf.get_variable('hash_param_%d' % (i),
                                [param_size[i],1],
                                dtype=tf.int64,
                                trainable=False,
                                initializer=tf.constant_initializer([0]))
        params.append(list(param)[0])

  orig_fids = fids
  fids, idx = tf.unique(fids, out_idx=tf.dtypes.int64)
  ret = None
  np = len(params)
  if np == 1:
    with tf.colocate_with(params[0]):
      ret = euler_hash_fid_v2(params[0], fids, 0, erase=erase)
  else:
    flat_ids = tf.reshape(fids, [-1])
    original_indices = tf.range(tf.size(flat_ids))
    p_assignments = tf.abs(flat_ids) % np
    new_ids = flat_ids // np

    p_assignments = tf.cast(p_assignments, tf.dtypes.int32)
    gather_ids = tf.dynamic_partition(new_ids, p_assignments, np)
    pindices = tf.dynamic_partition(original_indices,
                                    p_assignments, np)
    partitioned_result = []
    start = 0
    for p in xrange(np):
      pids = gather_ids[p]
      with tf.colocate_with(params[p]):
        result = euler_hash_fid_v2(params[p], pids, start, erase=erase)
        start = start + param_size[p]
      partitioned_result.append(result)
    ret = tf.dynamic_stitch(pindices,
                            partitioned_result)
  ret = tf.reshape(ret, [-1])
  return tf.gather(ret, idx, axis=0)


def hash_fid(fids, hash_space, multiplier=1.2, partition=None, use_locking=False):
  with tf.variable_scope('hash_fids', reuse=tf.AUTO_REUSE):
    var_dim = int(hash_space*multiplier)
    if partition is None:
      v = tf.get_variable("hash_param", 
                          [var_dim, 2],
                          dtype=tf.int64,
                          trainable=False,
                          initializer=tf.constant_initializer([0]))
    else:
      v = tf.get_variable("hash_param", 
                          [var_dim, 2],
                          dtype=tf.int64,
                          trainable=False,
                          initializer=tf.constant_initializer([0]),
                          partitioner=tf.fixed_size_partitioner(partition))
  orig_fids = fids
  fids, idx = tf.unique(fids, out_idx=tf.dtypes.int32)

  params = v
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  elif not isinstance(params, list):
    params = [params]

  ret = None
  np = len(params)
  if np == 1:
    with tf.colocate_with(params[0]):
      ret = euler_hash_fid(params[0], fids, 0, hash_space-1, 
                  use_locking=use_locking)
  else:
    flat_ids = tf.reshape(fids, [-1])
    original_indices = tf.range(tf.size(flat_ids))
    p_assignments = tf.abs(flat_ids) % np
    new_ids = flat_ids // np
    sub_space = hash_space // np

    p_assignments = tf.cast(p_assignments, tf.dtypes.int32)
    gather_ids = tf.dynamic_partition(new_ids, p_assignments, np)
    pindices = tf.dynamic_partition(original_indices,
                                    p_assignments, np)
    partitioned_result = []
    start = 0
    for p in xrange(np):
      pids = gather_ids[p]
      with tf.colocate_with(params[p]):
        if p == np-1:
          end = hash_space - 1
        else:
          end = start + sub_space - 1
        result = euler_hash_fid(params[p], pids, start, end, 
                      use_locking=use_locking)
        start = start + sub_space
      partitioned_result.append(result)
    ret = tf.dynamic_stitch(pindices, 
                            partitioned_result)
  ret = tf.reshape(ret, [-1])
  return tf.gather(ret, idx, axis=0)

def normalize(tensor, ord="euclidean", axis=None, name=None):
  """Normalizes `tensor` along dimension `axis` using specified norm.
  This uses `tf.linalg.norm` to compute the norm along `axis`.
  This function can compute several different vector norms (the 1-norm, the
  Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0) and
  matrix norms (Frobenius, 1-norm, 2-norm and inf-norm).
  Args:
    tensor: `Tensor` of types `float32`, `float64`, `complex64`, `complex128`
    ord: Order of the norm. Supported values are `'fro'`, `'euclidean'`, `1`,
      `2`, `np.inf` and any positive real number yielding the corresponding
      p-norm. Default is `'euclidean'` which is equivalent to Frobenius norm if
      `tensor` is a matrix and equivalent to 2-norm for vectors.
      Some restrictions apply: a) The Frobenius norm `'fro'` is not defined for
        vectors, b) If axis is a 2-tuple (matrix norm), only `'euclidean'`,
        '`fro'`, `1`, `2`, `np.inf` are supported. See the description of `axis`
        on how to compute norms for a batch of vectors or matrices stored in a
        tensor.
    axis: If `axis` is `None` (the default), the input is considered a vector
      and a single vector norm is computed over the entire set of values in the
      tensor, i.e. `norm(tensor, ord=ord)` is equivalent to
      `norm(reshape(tensor, [-1]), ord=ord)`. If `axis` is a Python integer, the
      input is considered a batch of vectors, and `axis` determines the axis in
      `tensor` over which to compute vector norms. If `axis` is a 2-tuple of
      Python integers it is considered a batch of matrices and `axis` determines
      the axes in `tensor` over which to compute a matrix norm.
      Negative indices are supported. Example: If you are passing a tensor that
        can be either a matrix or a batch of matrices at runtime, pass
        `axis=[-2,-1]` instead of `axis=None` to make sure that matrix norms are
        computed.
    name: The name of the op.
  Returns:
    normalized: A normalized `Tensor` with the same shape as `tensor`.
    norm: The computed norms with the same shape and dtype `tensor` but the
      final axis is 1 instead. Same as running
      `tf.cast(tf.linalg.norm(tensor, ord, axis keepdims=True), tensor.dtype)`.
  Raises:
    ValueError: If `ord` or `axis` is invalid.
  """
  with tf.name_scope(name, "normalize", [tensor]) as name:
    tensor = tf.convert_to_tensor(tensor)
    norm = tf.linalg.norm(tensor, ord, axis, keepdims=True)
    norm = tf.cast(norm, tensor.dtype)
    normalized = tensor / norm
    return normalized, norm
