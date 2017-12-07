import tensorflow as tf
import numpy as np

import csv

input_file = "../../data/house_price_number.csv"

x_col = []
y_col = []
id_col = []
price_col = []

count = 5876

pos_np = np.zeros((count, 2), dtype=np.float32)
price_np = np.zeros((count, 1), dtype=np.float32)

index = 0
with open(input_file, 'rt') as ifile:
	reader = csv.DictReader(ifile)	
	for row in reader;
		id_col.append(row['id'])
		x_col.append(row['x'])
		y_col.append(row['y'])
		price_col.append(row['price'])
		pos_np[index] = [row['x'], row['y']]
		price_np[index] = row['price']
		index += 1
print 'pos_np shape is:' , pos_np.shape

xs = tf.placeholder(tf.float32, pos_np)
ys = tf.placeholder(tf.float32, price_np)

def add_layer(inputs, in_size, out_size, activation_function=None):
	weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
	Wx_plus_b = tf.matmul(inputs, weights) + biases

	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_funcation(Wx_plus_b)
	return outputs



