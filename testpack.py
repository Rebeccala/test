import os
import tensorflow as tf
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #只显示error和warining信息

tf.InteractiveSession()
a = tf.zeros((3, 3))
b = tf.ones((3, 3))

print(tf.reduce_sum(b, reduction_indices=1).eval())
print(a.get_shape())

a = np.zeros((3, 3))
b = np.ones((3, 3))
print(np.sum(b, axis=1))
print(a.shape)
