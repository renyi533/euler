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

import collections

import tensorflow as tf

from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python import metrics

ModelOutput = collections.namedtuple(
    'ModelOutput', ['embedding', 'loss', 'metric_name', 'metric'])


class Model(layers.Layer):
  """
  """

  def __init__(self, **kwargs):
    super(Model, self).__init__(**kwargs)
    self.batch_size_ratio = 1

class UnsupervisedModel(Model):
  """
  Base model for unsupervised network embedding model.
  """

  def __init__(self,
               node_type,
               edge_type,
               max_id,
               num_negs=5,
               loss_type='xent',
               share_negs=False,
               rr_reweight=False,
               switch_side=False,
               mrr_ema_ratio=0.999,
               **kwargs):
    super(UnsupervisedModel, self).__init__(**kwargs)
    self.node_type = node_type
    self.edge_type = edge_type
    self.max_id = max_id
    self.num_negs = num_negs
    self.loss_type = loss_type
    self.share_negs = share_negs
    self.switch_side = switch_side
    self.rr_reweight = rr_reweight
    self.mrr_ema_ratio = mrr_ema_ratio

  def to_sample(self, inputs):
    batch_size = tf.size(inputs)
    src = tf.expand_dims(inputs, -1)
    pos = euler_ops.sample_neighbor(inputs, self.edge_type, 1,
                                    self.max_id + 1)[0]
    negs = euler_ops.sample_node_with_src(tf.reshape(pos,[-1]),
                    self.num_negs, self.share_negs)
    negs = tf.reshape(negs, [batch_size, self.num_negs])
    return src, pos, negs

  def target_encoder(self, inputs):
    raise NotImplementedError()

  def context_encoder(self, inputs):
    raise NotImplementedError()

  def _mrr(self, aff, aff_neg):
    aff_all = tf.concat([aff_neg, aff], axis=2)
    size = tf.shape(aff_all)[2]
    _, indices_of_ranks = tf.nn.top_k(aff_all, k=size)
    _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
    rr = tf.reciprocal(tf.to_float(ranks[:, :, -1] + 1))
    with tf.variable_scope('mrr_scope', reuse=tf.AUTO_REUSE):
      mrr_var = tf.get_variable('mrr', shape=[], dtype=tf.float32, 
                  initializer=tf.constant_initializer(0),
                  trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    curr_mrr = tf.reduce_mean(rr)
    mrr_delta = mrr_var*self.mrr_ema_ratio + (1-self.mrr_ema_ratio)*curr_mrr - mrr_var
    mrr = tf.assign_add(mrr_var, mrr_delta)
    return mrr, tf.reshape(ranks[:, :, -1], [-1])

  def decoder(self, embedding, embedding_pos, embedding_negs):
    logits = tf.matmul(embedding, embedding_pos, transpose_b=True)
    neg_logits = tf.matmul(embedding, embedding_negs, transpose_b=True)
    mrr, ranks = self._mrr(logits, neg_logits)
    rr_weight = euler_ops.reciprocal_rank_weight(tf.reshape(ranks, [-1]))
    rr_weight = tf.stop_gradient(rr_weight)
    mean_rr_weight = tf.reduce_mean(rr_weight)
    rr_weight = tf.expand_dims(rr_weight, -1)
    rr_weight = tf.expand_dims(rr_weight, -1)

    if not self.rr_reweight:
      rr_weight = 1.0
      mean_rr_weight = 1.0
      print('disable reciprocal rank reweight')
    else:
      print('enable reciprocal rank reweight')
      
    if self.loss_type == 'xent':
      true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.ones_like(logits), logits=logits)
      true_xent = tf.multiply(true_xent, rr_weight) / mean_rr_weight
      negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.zeros_like(neg_logits), logits=neg_logits)
      negative_xent = tf.multiply(negative_xent, rr_weight) / mean_rr_weight  
      loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
    elif self.loss_type == 'margin':
      delta = neg_logits + 1 - logits
      delta = delta * rr_weight / mean_rr_weight
      loss = tf.reduce_sum(tf.maximum(delta, 0.0))
    else:
      assert self.loss_type == 'rank'
      all_logits = tf.concat([neg_logits, logits], axis=2)
      all_cost = tf.reduce_logsumexp(all_logits, axis=2, keepdims=True)
      delta = (logits - all_cost) * rr_weight / mean_rr_weight
      loss = -tf.reduce_sum(delta)
    return loss, mrr

  def call(self, inputs):
    src, pos, negs = self.to_sample(inputs)
    if self.switch_side:
      embedding = self.context_encoder(src)
      embedding_pos = self.target_encoder(pos)
    else:
      embedding = self.target_encoder(src)
      embedding_pos = self.context_encoder(pos)

    negs_1d = tf.reshape(negs, [-1])
    uniq_negs, idx = tf.unique(negs_1d)

    if self.switch_side:
      embedding_negs = self.target_encoder(uniq_negs)
    else:
      embedding_negs = self.context_encoder(uniq_negs)

    embedding_negs = tf.gather(embedding_negs, idx, axis=0)
    embedding_negs = tf.reshape(embedding_negs,
                                [tf.shape(embedding)[0],self.num_negs,-1]) 
    loss, mrr = self.decoder(embedding, embedding_pos, embedding_negs)
    if self.switch_side:
      print("switch target/context side within UnsupervisedModel")
      embedding = self.context_encoder(inputs)
    else:
      print("Not switch target/context side within UnsupervisedModel")
      embedding = self.target_encoder(inputs)
    return ModelOutput(
        embedding=embedding, loss=loss, metric_name='mrr', metric=mrr)

class SupervisedModel(Model):
  """
  Base model for supervised network embedding model.
  """

  def __init__(self,
               label_idx,
               label_dim,
               num_classes=None,
               sigmoid_loss=False,
               **kwargs):
    super(SupervisedModel, self).__init__()
    self.label_idx = label_idx
    self.label_dim = label_dim
    if num_classes is None:
      num_classes = label_dim
    if label_dim > 1 and label_dim != num_classes:
      raise ValueError('laben_dim must match num_classes.')
    self.num_classes = num_classes
    self.sigmoid_loss = sigmoid_loss

    self.predict_layer = layers.Dense(num_classes)

  def encoder(self, inputs):
    raise NotImplementedError()

  def decoder(self, embeddings, labels):
    logits = self.predict_layer(embeddings)
    if self.sigmoid_loss:
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
      predictions = tf.nn.sigmoid(logits)
      predictions = tf.floor(predictions + 0.5)
    else:
      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
      predictions = tf.nn.softmax(logits)
      predictions = tf.one_hot(
          tf.argmax(predictions, axis=1), self.num_classes)
    loss = tf.reduce_mean(loss)
    return predictions, loss

  def call(self, inputs):
    labels = euler_ops.get_dense_feature(inputs, [self.label_idx],
                                         [self.label_dim])[0]
    if self.label_dim == 1:
      labels = tf.one_hot(tf.to_int64(tf.squeeze(labels)), self.num_classes)

    embedding = self.encoder(inputs)
    predictions, loss = self.decoder(embedding, labels)
    f1 = metrics.f1_score(labels, predictions, name='f1')

    return ModelOutput(
        embedding=embedding, loss=loss, metric_name='f1', metric=f1)
