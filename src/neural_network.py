
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
count = 5884

dis_np = np.zeros((count, 1), dtype=np.float32)
price_np = np.zeros((count, 1), dtype=np.float32)
info_np = np.zeros((count, 2), dtype=np.float32)

index = 0
csv_file = "../data/price.csv"

with open(csv_file, 'rt', encoding='utf-8') as ifile:
    reader = csv.DictReader(ifile)
    rows = [row for row in reader]

    for row in rows:
        dis = float(row['distance'])
        price = float(row['price']) / max
        dis_np[index] = [dis]
        price_np[index] = [price]
        info_np[index] = [dis, price]
        index += 1

print(info_np.shape)
print(info_np[0])

dis_train = torch.from_numpy(dis_np)
price_train = torch.from_numpy(price_np)
info_train = torch.from_numpy(info_np)


# 定义简单的前馈神经网络
class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


model = Neuralnetwork(1, 300, 100, 1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    running_loss = 0.0
    running_acc = 0.0
    inputs = Variable(dis_train)
    target = Variable(price_train)

    # 向前传播
    out = model(inputs)
    loss = criterion(out, target)
    # 向后传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'
              .format(epoch+1, num_epoches, loss.data[0]))

model.eval()
predict = model(Variable(dis_train))
predict = predict.data.numpy()

plt.plot(dis_train.numpy(), price_train.numpy(), 'ro', label='Original data')
plt.plot(dis_train.numpy(), predict, label='Fitting Line')
plt.show()

# 保存模型
#torch.save(model.state_dict(), './neural_network.pth')