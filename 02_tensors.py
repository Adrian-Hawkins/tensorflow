import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Immutable
# x = tf.constant(4, shape=(1,1), dtype=tf.float32)
# x = tf.constant([[1,2,3],[4,5,6]])
# x = tf.zeros((3,3))
# x = tf.eye(3)
# x = tf.random.normal((3,3), mean=0, stddev=1)
# x = tf.random.uniform((3,3), minval=0, maxval=1)
# x = tf.range(10)
# print(x)

# Mutable cast
# x = tf.cast(x, dtype=tf.float32)
# print(x)

# Mathematical operations
# x = tf.constant([1,2,3])
# y = tf.constant([4,5,6])
# z = tf.add(x,y)
# z = x+y
# z= tf.subtract(x,y)
# z = x-y
# z = tf.divide(x,y)
# z = x / y
# z = tf.multiply(x,y)
# z = x*y
# z= tf.tensordot(x,y, axes=1)
# z = x ** 3 # Every element to the power of 3
# print(z)

# x = tf.random.normal((2,3))
# y = tf.random.normal((3,4))

# z = tf.matmul(x,y) # row x must match col y
# z = x @ y same as mat mul???

# print(z)

# x = tf.constant([[1,2,3,4], [5,6,7,8]])
# print(x[0, :]) # First row all cols
# print(x[:, 0]) # First col all rows
# print(x[0:2, 1:4])

# x = tf.random.normal((2,3))
# print(x)
# reshape
# x = tf.reshape(x, (3,2))
# print(x)

# x = tf.reshape(x, (-1,2))
# print(x)
# x = x.numpy()
# print(x)
# print(type(x))

# x = tf.convert_to_tensor(x)
# print(type(x))

# x = tf.constant(["Adrian"])
# print(x)

x = tf.constant([1,2,3])
print(x)
x =  tf.Variable([1,2,3])
print(x)