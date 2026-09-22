import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
with tf.device("/GPU:0"):
    c = tf.matmul(a, b)
print("Result:")
print(c.numpy())
print("Executed on:", c.device)
