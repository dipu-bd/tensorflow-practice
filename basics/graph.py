import tensorflow as tf

a = tf.placeholder(tf.float32, name='a')
b = tf.placeholder(tf.float32, name='b')
x = tf.add(a, b, name='add_a_b')
y = tf.multiply(a, b, name='mul_a_b')
z = tf.add(x, y, name='add_x_y')

sess = tf.Session()
writer = tf.summary.FileWriter('.bin', sess.graph)

res = sess.run(z, {a: 3, b: 5})
print('\nResult:', res)

writer.close()
sess.close()
