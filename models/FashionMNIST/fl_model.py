import load_data
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
# Training settings
log_interval = 10

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device (  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')


class Generator(load_data.Generator):
    """Generator for FashionMNIST dataset."""

    # Extract FashionMNIST data using torchvision datasets
    def read(self, path, testset_device = 'cpu'):
        transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
        self.trainset = datasets.FashionMNIST(
            path, train=True, download=True, transform=transf)
        self.testset = datasets.FashionMNIST(
            path, train=False, transform=transf)
        self.labels = list(self.trainset.classes)
        if testset_device == 'cuda':
            x = torch.Tensor(self.testset.data/255).unsqueeze(1)
            self.testset.data = transf.transforms[-1](x)
            self.testset = TensorDataset(torch.tensor(self.testset.data).to(testset_device), torch.tensor(self.testset.targets).to(testset_device))


# #LeNet
from torch.nn import Module
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


def get_optimizer(model, local_lr, local_momentum, weight_decay):
    return optim.SGD(model.parameters(), lr=local_lr, momentum=local_momentum, weight_decay=weight_decay)


def get_trainloader(trainset, batch_size):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)


def get_testloader(testset, batch_size):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

def extract_weights(model, dev = 'cpu'):
    weights = []
    for name, weight in model.to(torch.device(dev)).named_parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            weights.append((name, weight.data))

    return weights


def load_weights(model, weights):
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)


def train(model, trainloader, optimizer, local_iter, epochs):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    tau = 0 
    if epochs:
        for ep in range(local_iter):
            for data in trainloader:
                tau += 1
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                train_loss += loss.item()*inputs.shape[0]
                loss.backward()
                optimizer.step()
    else:
        for iter_ in range(1, local_iter + 1):
            tau += 1
            data = next(iter(trainloader))
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()*inputs.shape[0]
            loss.backward()
            optimizer.step()
    model.eval()
    return model, tau, train_loss/tau

def test(model, testloader):
    model.to(device)
    model.eval()

    with torch.no_grad():
        criterion = nn.CrossEntropyLoss(reduction='sum')
        correct = 0
        total = 0
        loss_ = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_ += float(criterion(outputs, labels))
            predicted = torch.argmax(  # pylint: disable=no-member
                outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy, loss_/testloader.__len__()
