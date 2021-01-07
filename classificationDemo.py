import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data


#

# print(x.shape, y.shape)
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0)
# plt.show()
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)
BATCH_SIZE = 20
if __name__ == '__main__':
    # fake data
    n_data = torch.ones(100, 2)
    x0 = torch.normal(2 * n_data, 1)
    y0 = torch.zeros(100)
    x1 = torch.normal(-2 * n_data, 1)
    y1 = torch.ones(100)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    y = torch.cat((y0, y1), ).type(torch.LongTensor)
    torch_dataset = Data.TensorDataset(x,y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    # net = Net(n_feature=2, n_hidden=10, n_output=2)
    net = net2
    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            out = net(batch_x)
            loss = loss_func(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 2 == 0:
                prediction = torch.max(F.softmax(out,dim=1), 1)[1]
                pred_y = prediction.data.numpy().squeeze()
                target_y = batch_y.data.numpy()
                accuracy = sum(pred_y == target_y) / BATCH_SIZE
                print("epoch:{},step:{},accuracy:{}".format(epoch+1,step+1,accuracy))
        # for t in range(100):
        #     out = net(x)
        #     loss = loss_func(out, y)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
    torch.save(net, 'net.pkl')
    torch.save(net.state_dict(), 'net_params.pkl')
        # if t % 2 == 0:
        #     plt.cla()
        #     prediction = torch.max(F.softmax(out), 1)[1]
        #     pred_y = prediction.data.numpy().squeeze()
        #     target_y = y.data.numpy()
        #     plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0)
        #     accuracy = sum(pred_y == target_y) / 200
        #     plt.text(1.5, -4, 'Accuary=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        #     plt.ioff()
        #     plt.show()
