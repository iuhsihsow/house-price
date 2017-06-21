
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
record_count = 0
x_center = 116.407718
y_center = 38.915599

max = 154953.0
count = 5876

price_np = np.zeros((count, 1), dtype=np.float32)
dis_np = np.zeros((count, 1), dtype=np.float32)
index_np = np.zeros((count, 1), dtype=np.float32)
info_np = np.zeros((count, 5), dtype=np.float32)

index = 0
csv_file = "../data/house_price_number.csv"

with open(csv_file, 'rt', encoding='utf-8') as ifile:
    reader = csv.DictReader(ifile)
    for row in reader:
        vip_school = float(row['n_school'])
        b_subway = float(row['b_subway'])
        dis = float(row['distance'])
        green_rate = float(row['greening_rate']) / 100.0
        plot_area = float(row['plot_area'])
        price = float(row['price']) / max
        info_np[index] = [vip_school, b_subway, dis, green_rate, plot_area]
        price_np[index] = [price]
        index_np[index] = index
        dis_np[index] = [dis]
        index += 1

print(info_np.shape)
print(info_np[0])

info_train = torch.from_numpy(info_np)
price_train = torch.from_numpy(price_np)


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

num_epoches = 10000
learning_rate = 1e-5
model = Neuralnetwork(5 , 300, 10, 1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    running_loss = 0.0
    running_acc = 0.0
    inputs = Variable(info_train)
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
predict = model(Variable(info_train))
predict = predict.data.numpy()

np.savetxt("../data/predict.csv", predict * max)

def calc_mape(pred, org, size):
    sum = 0
    for i in range(count):
        if price_np[i][0] <= 0.0001:
            print('error')
        sum += (abs(pred[i][0] - price_np[i][0])) / price_np[i][0]
    sum /= size
    return sum

print(calc_mape(predict, price_np, count))


#plt.plot(info_train.numpy(), price_train.numpy(), 'ro', label='Original data')
#plt.plot(info_train.numpy(), predict, label='Fitting Line')
#plt.plot(dis_np, price_np, label='Fitting Line')
#plt.plot(dis_np, abs(price_np - predict)/price_np , label='Fitting Line')
plt.plot(index_np, abs(price_np - predict)/price_np , label='Fitting Line')

plt.show()

# 保存模型
#torch.save(model.state_dict(), './neural_network.pth')