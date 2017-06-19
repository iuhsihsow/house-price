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

with open(csv_file, 'rt', encoding='utf-8') as ifile:
    reader = csv.DictReader(ifile)
    rows = [row for row in reader]
    for row in rows:
        if record_count % 3 != 0:
            id_col.append(row['id'])
            x_col.append(float(row[x_field]))
            y_col.append(float(row[y_field]))
            id_col.append(int(row[id_field]))
            dis = (float(row[x_field]) - 116.407718) * (float(row[x_field]) - 116.407718) \
                  + (float(row[y_field]) - 38.915599) * (float(row[y_field]) - 38.915599)
            dis_col.append([dis, 0])

            price_col.append(float(row[price_field]))
            pos_col.append([float(row[x_field]),float(row[y_field])])
        record_count += 1

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
H = 1000   # set myself
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

learning_rate = 1e-6

for t in range(500):
  y_pred = x.mm(w1).clamp(min=0).mm(w2)
  loss = (y_pred - y).pow(2).sum()
  print(t, loss.data[0])

  loss.backward()

  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data

  # Manually zero the gradients before running the backward pass
  w1.grad.data.zero_()
  w2.grad.data.zero_()