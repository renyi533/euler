from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_euler.python import encoders
from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python.models import base

class SignedUnsupervisedModel(base.Model):
  """
  Base model for unsupervised network embedding model.
  """
  def __init__(self,
               node_type,
               edge_type,
               neg_edge_type,
               max_id,
               context_max_id,
               num_negs=5,
               loss_type='xent',
               share_negs=False,
               *args,
               **kwargs):
    super(SignedUnsupervisedModel, self).__init__(*args, **kwargs)
    self.node_type = node_type
    self.edge_type = edge_type
    self.neg_edge_type = neg_edge_type
    self.max_id = max_id
    self.context_max_id = context_max_id
    self.num_negs = num_negs
    self.loss_type = loss_type
    self.share_negs = share_negs

  def to_sample(self, inputs):
    batch_size = tf.size(inputs)
    src = tf.expand_dims(inputs, -1)
    pos = euler_ops.sample_neighbor(inputs, self.edge_type, 1,
                                    self.max_id + 1)[0]
    pos_negs = euler_ops.sample_node_with_src(tf.reshape(pos,[-1]),
                    self.num_negs, self.share_negs)                    
    pos_negs = tf.reshape(pos_negs, [batch_size, self.num_negs])

    neg = euler_ops.sample_neighbor(inputs, self.neg_edge_type, 1,
                                    self.max_id + 1)[0]
    neg_negs = euler_ops.sample_node_with_src(tf.reshape(neg,[-1]),
                    self.num_negs, self.share_negs)                    
    neg_negs = tf.reshape(neg_negs, [batch_size, self.num_negs])

    return src, pos, pos_negs, neg, neg_negs

  def target_encoder(self, inputs):
    raise NotImplementedError()

  def context_encoder(self, inputs):
    raise NotImplementedError()

  def _mrr(self, aff, aff_neg):
    aff_all = tf.concat([aff_neg, aff], axis=2)
    size = tf.shape(aff_all)[2]
    _, indices_of_ranks = tf.nn.top_k(aff_all, k=size)
    _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
    return tf.reduce_mean(tf.reciprocal(tf.to_float(ranks[:, :, -1] + 1)))

  def decoder(self, embedding, embedding_pos, embedding_negs):
    logits = tf.matmul(embedding, embedding_pos, transpose_b=True)
    neg_logits = tf.matmul(embedding, embedding_negs, transpose_b=True)
    mrr = self._mrr(logits, neg_logits)
    if self.loss_type == 'xent':
      true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.ones_like(logits), logits=logits)
      negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.zeros_like(neg_logits), logits=neg_logits)
      loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
    elif self.loss_type == 'margin':
      delta = neg_logits + 1 - logits
      loss = tf.reduce_sum(tf.maximum(delta, 0.0))
    else:
      assert self.loss_type == 'rank'
      all_logits = tf.concat([neg_logits, logits], axis=2)
      all_cost = tf.reduce_logsumexp(all_logits, axis=2, keepdims=True)
      loss = -tf.reduce_sum(logits - all_cost)
    return loss, mrr

  def call(self, inputs):
    src, pos, pos_negs, neg, neg_negs = self.to_sample(inputs)
    embedding = self.target_encoder(src)
    embedding_pos = self.context_encoder(pos)
    embedding_pos_negs = self.context_encoder(pos_negs)
    embedding_neg = -self.context_encoder(neg)
    embedding_neg_negs = -self.context_encoder(neg_negs)
    
    embedding = tf.concat([embedding, embedding], axis=0)
    embedding_pos = tf.concat([embedding_pos, embedding_neg], axis=0)
    embedding_negs = tf.concat([embedding_pos_negs, embedding_neg_negs], axis=0)

    loss, mrr = self.decoder(embedding, embedding_pos, embedding_negs)
    embedding = self.target_encoder(inputs)
    return base.ModelOutput(
        embedding=embedding, loss=loss, metric_name='mrr', metric=mrr)

