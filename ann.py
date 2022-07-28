import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

def readData(batch_size = 64):
    # transform = transforms.Compose([
    #     transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    # ])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root='../dataset/mnist/',
                                   train=True,
                                   download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size)
    test_dataset = datasets.MNIST(root='../dataset/mnist/',
                                  train=False,
                                  download=True,
                                  transform=transform)
    test_loader = DataLoader(test_dataset,
                             shuffle=True,
                             batch_size=batch_size)
    return train_loader, test_loader

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(784, 10, bias=False)
        # self.layer = torch.nn.Sequential(
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(784, 10, bias=False),
        #     torch.nn.ReLU())
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc(x))
        # x = self.layer(x)
        return x

def train(train_loader, epoch, binary = True):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        if binary:
            inputs = torch.where(inputs > 0.1307, torch.tensor([1.]), torch.tensor([0.]))
        target = F.one_hot(target).float()*10
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss:: %.3f' % (epoch+1, batch_idx+1, running_loss))
            running_loss = 0

def test(test_loader, binary = True):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if binary:
                images = torch.where(images > 0.1307, torch.tensor([1.]), torch.tensor([0.]))
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy on test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    train_loader, test_loader = readData(128)
    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(20):
        train(train_loader, epoch, False)
        test(test_loader, False)
    torch.save(model.state_dict(), 'weights.pt')