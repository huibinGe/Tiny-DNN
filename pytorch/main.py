import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

def LoadData(csv_file):

    with open(csv_file, 'r') as f:
        inputs, labels = [], []
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue
            data = line.strip().split(';')
            x = [float(x_i) for x_i in data[:-1] ]
            y = 1 if float(data[-1])> 5 else -1
            inputs.append(x)
            labels.append([y])
    f.close()
    return torch.Tensor(inputs), torch.Tensor(labels)

class Preceptron(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Preceptron, self).__init__()
        self.layer = nn.Linear(input_dims, output_dims, bias=False)
    def forward(self, x):
        return self.layer(x)
def train(inputs, labels):
    input_dims = inputs.shape[1]
    out_dims = labels.shape[1]
    net = Preceptron(input_dims, out_dims)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    best_acc = 0
    for epoch in range(500):
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        right = 0
        total = 0
        for out, label in zip(outputs.squeeze().tolist(), labels.squeeze().tolist()):
            predict = 1 if out > 0.0 else -1
            if predict == label:
                right += 1
            total += 1
        acc = right / total
        if acc > best_acc:
            best_acc = acc
        print("BEST accuary of epoch {} is {}".format(epoch, best_acc))




def main(csv_file):
    inputs, labels = LoadData(csv_file)
    train(inputs, labels)



if __name__ == '__main__':
    csv_file = "../data/winequality-white.csv"
    main(csv_file)

