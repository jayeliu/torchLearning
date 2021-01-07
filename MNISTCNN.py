import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda() / 255.
test_y = test_data.test_labels[:2000].cuda()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # in:1,28,28
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # hid:16,14,14
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # hid:32,7,7
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平（batch_size,32*7*7）
        output = self.out(x)
        return output


if __name__ == '__main__':
    cnn = CNN().cuda()
    # print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = x.cuda()
            b_y = y.cuda()
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print("epoch:%4d,step:%4d,train loss:%.4f" % (epoch, step, loss))
