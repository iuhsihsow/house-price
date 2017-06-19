import torch
from torch.autograd import Variable

import csv

csv_file = "../data/beijing_x_y_price.csv"

x_field = 'X'
y_field = 'Y'
id_field = 'id'
price_field = 'price'

x_col = []
y_col = []
id_col = []
price_col = []
dis_col = []

pos_col = []

record_count = 0
x_center = 116.407718
y_center = 38.915599

max_delta_x = 0
max_delta_y = 0

with open(csv_file, 'rt', encoding='utf-8') as ifile:
    reader = csv.DictReader(ifile)
    rows = [row for row in reader]
    for row in rows:
        if record_count % 3 != 0 and float(row[x_field]) > 100:
            id_col.append(row['id'])
            lon = float(row[x_field])
            lat = float(row[y_field])
            delta_x = abs(lon - x_center)
            delta_y = abs(lat - y_center)

            max_delta_x = max(max_delta_x, delta_x)
            max_delta_y = max(max_delta_y, delta_y)


            x_col.append(delta_x)
            y_col.append(delta_y)


            id_col.append(int(row[id_field]))

            dis = (float(row[x_field]) - 116.407718) * (float(row[x_field]) - 116.407718) \
                  + (float(row[y_field]) - 38.915599) * (float(row[y_field]) - 38.915599)
            dis_col.append([dis, 0])

            price_col.append(float(row[price_field]))
            pos_col.append([delta_x, delta_y])

        record_count += 1

    print(max_delta_x, max_delta_y)

price_norm = [float(i) / max(price_col) for i in price_col]
train_pos = torch.FloatTensor(pos_col)
train_price = torch.FloatTensor(price_norm)

print(train_pos)

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N = len(x_col)
D_in = 2  # lon and lat
H = 50   # set myself
D_out = 1 # just price

# Create random Tensors to hold input and outputs, and wrap them in Variables.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Variables during the backward pass.
x = Variable(train_pos.type(dtype), requires_grad=False)
y = Variable(train_price.type(dtype), requires_grad=False)

#x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
#y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-5

#for t in range(1000):
loss = 1
t = 0;
while(loss > 0.0001):
  y_pred = x.mm(w1).clamp(min=0).mm(w2)
  loss = (y_pred - y).pow(2).sum()
  if t % 1000 == 0:
    print(t, loss.data[0])
  t += 1

  loss.backward()

  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data

  # Manually zero the gradients before running the backward pass
  w1.grad.data.zero_()
  w2.grad.data.zero_()

print(t)