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
info_np = np.zeros((count, 2), dtype=np.float32)


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
            info_np[index] = [dis, price]
            index += 1

        record_count += 1
    #print(max_delta_x, max_delta_y)

print(info_np.shape)
print(info_np[0])

dis_train = torch.from_numpy(dis_np)
price_train = torch.from_numpy(price_np)
info_train = torch.from_numpy(info_np)
label_np = np.zeros((count, 10), dtype=np.float32)
label_train = torch.from_numpy(label_np)

class Logstic_Regression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Logstic_Regression, self).__init__()
        self.logstic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        out = self.logstic(x)
        return out

model = Logstic_Regression(2, 10)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)


# 开始训练
num_epochs = 4000
for epoch in range(num_epochs):
    inputs = Variable(info_train)
    label = Variable(label_train)
    running_loss = 0.0
    running_acc = 0.0

    # forward
    out = model(inputs)
    loss = criterion(out, label)

   # 向后传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




model.eval()
predict = model(Variable(dis_train))
predict = predict.data.numpy()

print(label)

plt.plot(dis_train.numpy(), label.numpy(), 'ro', label='Original data')
#plt.plot(dis_train.numpy(), predict, label='Fitting Line')
plt.show()

# 保存模型
#torch.save(model.state_dict(), './linear.pth')

# 结论是不能拟合