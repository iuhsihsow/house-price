
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
import torch.nn.functional as F

batch_size = 32
learning_rate = 0.01
num_epoches = 10000

input_file = "../data/house_price_number.csv"
output_file = "../data/high_accuracy_location.csv"

x_center = 116.407718
y_center = 38.915599
max_price = 154953.0
count = 5876

location_np = np.zeros((count, 2), dtype=np.float32)
index_np = np.zeros((count,1), dtype=np.uint32)
price_np = np.zeros((count, 1), dtype=np.float32)

index = 0
with open(input_file, 'rt') as ifile:
    reader = csv.DictReader(ifile)
    for row in reader:
        location_np[index] = [float(row['X']) - x_center, float(row['Y']) - y_center]
        price_np[index] = float(row['price']) / max_price
        index_np[index] = int(row['id'])
        index += 1

location_data = torch.from_numpy(location_np)
price_data = torch.from_numpy(price_np)


# 定义简单的前馈神经网络
class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, out_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))      # activation function for hidden layer
        x = self.layer2(x)             # linear output
        return x


model = NeuralNetwork(2, 30, 1)

loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    running_loss = 0.0
    running_acc = 0.0

    inputs = Variable(location_data)
    target = Variable(price_data)

    # 向前传播
    out = model(inputs)
    loss = loss_func(out, target)
    # 向后传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'
              .format(epoch+1, num_epoches, loss.data[0]))

model.eval()

predict = model(Variable(location_data))
predict = predict.data.numpy()

# output error<0.1
error = 10000
with open(output_file, 'w') as ofile:
    field_names = ['id', 'X', 'Y', 'Price', 'Predict Price']
    writer = csv.DictWriter(ofile, fieldnames=field_names)
    writer.writeheader()
    for x in range(0, count):
        if (abs(predict[x] - price_np[x]) / price_np[x]) < error:
            row_dict = {}
            row_dict['id'] = index_np[x]
            row_dict['X'] = location_np[x][0] + x_center
            row_dict['Y'] = location_np[x][1] + y_center
            row_dict['Price'] = price_np[x][0] * max_price
            row_dict['Predict Price'] = predict[x][0] * max_price
            writer.writerow(row_dict)

plt.scatter(price_np * max_price, abs(predict - price_np) * max_price, label='Scatter')
plt.show()

# 保存模型
#torch.save(model.state_dict(), './neural_network.pth')