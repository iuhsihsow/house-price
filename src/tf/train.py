import tensorflow as tf
import numpy as np

import csv
import matplotlib.pyplot as plt

input_file = "../../data/house_price_number.csv"

x_col = []
y_col = []
id_col = []
price_col = []
x_center = 116.407718
y_center = 38.915599
pmax = 154953.0

count = 5876

pos_np = np.zeros((count, 2), dtype=np.float32)
index_np = np.zeros((count,1), dtype=np.float32)
price_np = np.zeros((count, 1), dtype=np.float32)

index = 0
with open(input_file, 'rt') as ifile:
	reader = csv.DictReader(ifile)	
	for row in reader:
		pos_np[index] = [float(row['X']) - x_center, float(row['Y']) - y_center]
		price_np[index] = float(row['price'])/pmax
		index_np[index]=index
		index += 1
print 'pos_np shape is:' , pos_np.shape

xs = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, 1])

def add_layer(inputs, in_size, out_size, activation_function=None):
	weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
	Wx_plus_b = tf.matmul(inputs, weights) + biases

	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs


cluster = 50
l1 = add_layer(xs, 2, cluster, activation_function=tf.nn.relu)
prediction = add_layer(l1, cluster, 1, activation_function=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))


optimizer = tf.train.GradientDescentOptimizer(0.05)
train_step = optimizer.minimize(loss)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

for i in range(1000):
	sess.run(train_step, feed_dict={xs: pos_np, ys: price_np})
	if i % 10 == 0:
		print sess.run(loss, feed_dict={xs: pos_np, ys: price_np})

save_path = saver.save(sess, "../../data/tf.ckpt")
print "save to ", save_path

feed_dict = {xs: pos_np}
res = sess.run(prediction, feed_dict)

rate = 0.1

i = 0
for x in range(0, count):
	if (abs(res[x] - price_np[x]) / price_np[x]) < rate:
		i += 1
print 'good result:', i 

show_np = np.zeros((i, 2), dtype=np.float32)
i = 0
for x in range(0, count):
	if (abs(res[x] - price_np[x]) / price_np[x]) < rate:
		show_np[i] = [price_np[x], (abs(res[x] - price_np[x]) / price_np[x])]
		i += 1

print 'Mean:', np.mean(show_np, axis=0)
plt.scatter(show_np[:, 0] * pmax, show_np[:, 1], label='Fitting Line')
#plt.scatter(price_np * pmax, abs(price_np - res)/price_np, label='Fitting Line')
plt.show()

