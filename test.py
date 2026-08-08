import tensorflow as tf

N = 100

stack_a = tf.random.normal(shape=(N,), dtype=tf.float32)
stack_b = tf.random.normal(shape=(N,), dtype=tf.float32)
print(stack_a.ndim)
rand_a = tf.zeros(shape=(N, 2), dtype=tf.float32)
rand_b = tf.ones(shape=(N, 2), dtype=tf.float32)
print(rand_a.ndim)


concat = tf.stack([rand_a, rand_b], axis=0)
print(concat)

stack = tf.stack([stack_a, stack_b], axis=1)
print(stack)

inp = tf.keras.Input(shape=(10,), name="physical_inputs")

print(inp.ndim)

zeros = tf.zeros(shape=(N,))
ones = tf.ones(shape=(N,))
zeros_ones = tf.stack([zeros, ones], axis=1)

print(f"Stacked: {zeros_ones}")


reduce_sum = tf.reduce_sum(zeros_ones, axis=0)
print(reduce_sum)
