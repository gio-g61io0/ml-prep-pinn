import numpy as np
import tensorflow as tf

output_matrix = np.random.rand(128, 12288)
value_up_matrix = np.random.rand(12288, 12288)

one_d_output_matrix = np.random.rand(16,)
one_d_value_up_matrix = np.random.rand(16,)

res = np.dot(output_matrix, value_up_matrix)
one_d_res = np.multiply(one_d_value_up_matrix, one_d_output_matrix)
print(res.shape) #O
print(one_d_res.shape) #O



rand = tf.random.normal([128, 12288])
reduced = tf.reduce_sum(rand, axis=1)
print(reduced.shape)


input_batch = tf.random.uniform(shape=[128, 1], minval=1, maxval=10, dtype=tf.int64)
input_batch = tf.reshape(input_batch, [ -1])
one_hot = tf.one_hot(input_batch, depth=10)
print(input_batch)
print(f"One hot shape:{one_hot.shape}")
print(one_hot)

k_per_sample = tf.reduce_sum(one_hot * 10, axis=0)
print(k_per_sample)
print(k_per_sample.shape)
