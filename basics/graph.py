import tensorflow as tf

a = tf.constant(3, name='A')
b = tf.constant(5, name='B')
x = tf.add(a, b)
y = tf.multiply(a, b)
z = tf.add(x, y)

sess = tf.Session()
writer = tf.summary.FileWriter('./../.bin', sess.graph)

res = sess.run(z)
print('\nResult:', res)

writer.close()
sess.close()
