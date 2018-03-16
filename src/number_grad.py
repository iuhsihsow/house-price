
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import csv
import matplotlib.pyplot as plt

batch_size = 32
learning_rate = 1e-2
num_epoches = 1000
record_count = 0
x_center = 116.407718
y_center = 38.915599

max = 154953.0
count = 5876

price_np = np.zeros((count, 1), dtype=np.float32)
info_np = np.zeros((count, 5), dtype=np.float32)

index = 0
csv_file = "../data/house_price_number.csv"

with open(csv_file, 'rt', encoding='utf-8') as ifile:
    reader = csv.DictReader(ifile)
    for row in reader:
        vip_school = float(row['n_school']) + 1.0
        b_subway = float(row['b_subway']) + 1.0
        dis = float(row['distance'])
        green_rate = float(row['greening_rate']) / 100.0
        plot_area = float(row['plot_area'])
        price = float(row['price']) / max
        info_np[index] = [vip_school, b_subway, dis, green_rate, plot_area]
        price_np[index] = [price]
        index += 1

print(info_np.shape)
print(info_np[0])

info_train = torch.from_numpy(info_np)
price_train = torch.from_numpy(price_np)

dtype = torch.FloatTensor
# Create random Tensors to hold input and outputs, and wrap them in Variables.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Variables during the backward pass.
#x = Variable(info_train.type(dtype), requires_grad=False)
#y = Variable(price_train.type(dtype), requires_grad=False)

N = count
D_in = 5
D_out = 1
Mid = 200

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.
w1 = Variable(torch.randn(D_in, Mid).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(Mid, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6

t = 0;
#while(loss_value > 0.0001):
for t in range(5000):
  y_pred = x.mm(w1).clamp(min=0).mm(w2)
  loss = (y_pred - y).pow(2).sum()
  loss_value = loss.data[0]
  if t % 100 == 0:
    print(t, loss.data[0])
  t += 1

  loss.backward()

  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data

  # Manually zero the gradients before running the backward pass
  w1.grad.data.zero_()
  w2.grad.data.zero_()

print(y_pred)
print(y_pred.data.numpy())
y_pred = x.mm(w1).clamp(min=0).mm(w2)

sum = 0
for i in range(count):
    if price_np[i][0] <= 0.0001:
        print('error')
    sum += (y_pred.data.numpy()[i][0] - price_np[i][0]) / price_np[i][0]

sum = sum / count


print(sum)

print(y_pred)

