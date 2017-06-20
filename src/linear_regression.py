import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import csv

csv_file = "../data/price.csv"

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

max_delta_x = 1.6952936584449958
max_delta_y = 2.2264053110386968
max = 154953.0

count = 5884

dis_np = np.zeros((count, 1), dtype=np.float32)
price_np = np.zeros((count, 1), dtype=np.float32)

index = 0

with open(csv_file, 'rt', encoding='utf-8') as ifile:
    reader = csv.DictReader(ifile)
    rows = [row for row in reader]

    for row in rows:
        if True:
            id_col.append(row['id'])
            dis = float(row['distance'])
            price = float(row['price']) / max
            dis_np[index] = [dis]
            price_np[index] = [price]
            index += 1

        record_count += 1
    #print(max_delta_x, max_delta_y)


dis_train = torch.from_numpy(dis_np)
price_train = torch.from_numpy(price_np)

# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)


# 开始训练
num_epochs = 4000000
for epoch in range(num_epochs):
    inputs = Variable(dis_train)
    target = Variable(price_train)

    # forward
    out = model(inputs)
    loss = criterion(out, target)
    if loss.data[0] < 0.035:
        break
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'
              .format(epoch+1, num_epochs, loss.data[0]))

model.eval()
predict = model(Variable(dis_train))
predict = predict.data.numpy()

plt.plot(dis_train.numpy(), price_train.numpy(), 'ro', label='Original data')
plt.plot(dis_train.numpy(), predict, label='Fitting Line')
plt.show()

# 保存模型
#torch.save(model.state_dict(), './linear.pth')

# 结论是不能拟合