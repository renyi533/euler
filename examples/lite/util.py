import tensorflow as tf

def get_top_neighbor(path, source, default_id, k, skip_self=True):
  def _iter_body(i, state):
    path, source, default_id, ta = state
    y, idx, count = tf.unique_with_counts(path[i], out_idx=tf.int64)

    zero_count = tf.zeros_like(count, dtype=tf.int64)

    if skip_self:
      mask = tf.equal(y, source[i])
      count = tf.where(mask, zero_count, count)
    
    mask = tf.equal(y, tf.to_int64(default_id))
    count = tf.where(mask, zero_count, count)
    
    sort_idx = tf.argsort(count, direction='DESCENDING')
    sorted_neighbor = tf.gather(y, sort_idx)
    n_k = tf.minimum(tf.shape(sorted_neighbor)[0], k)
    top_slice =  tf.slice(sorted_neighbor, [0], [n_k])
    top_slice = tf.cond(n_k < k, 
                        lambda: tf.concat([top_slice, tf.zeros([k-n_k], dtype=tf.int64)], axis=0), 
                        lambda: top_slice)

    sorted_count = tf.gather(count, sort_idx)
    count_slice =  tf.slice(sorted_count, [0], [n_k])
    count_slice = tf.cond(n_k < k, 
                          lambda: tf.concat([count_slice, tf.zeros([k-n_k], dtype=tf.int64)], axis=0), 
                          lambda: count_slice)
    
    top_slice = tf.stack([top_slice, count_slice], axis=0)

    out_ta = ta.write(i, tf.reshape(top_slice, [1, 2, k]))
    return i+1, (path, source, default_id, out_ta)
  
  rows = tf.shape(path)[0]
  ta = tf.TensorArray(dtype=tf.int64, size=rows, infer_shape=False)
  init_state = (0, (path, source, default_id, ta))
  condition = lambda i, _: i < rows
  _, (_, _, _, ta_final) = tf.while_loop(condition, _iter_body, init_state)
  tensor_final = ta_final.concat()
  return tensor_final

