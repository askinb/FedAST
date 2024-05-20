import load_data
import logging
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import math
import torchvision
import torchvision.models as models

# Training settings

log_interval = 10

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device(  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')


class Generator(load_data.Generator):
    """Generator for CIFAR-10 dataset."""

    # Extract CIFAR-10 data using torchvision datasets
    def read(self, path, testset_device = 'cpu'):
        transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.trainset = datasets.CIFAR10(
            path, train=True, download=True, transform=transf)
        self.testset = datasets.CIFAR10(
            path, train=False, transform=transf)
        self.labels = list(self.trainset.classes)
        if testset_device == 'cuda':
            x = torch.Tensor(self.testset.data/255).permute(0,3,1,2)
            self.testset.data = transf.transforms[-1](x)
            self.testset = TensorDataset(torch.tensor(self.testset.data).to(testset_device), torch.tensor(self.testset.targets).to(testset_device))
    
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo




#FedDyn CIFAR-10 model
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*5*5, 384) 
        self.fc2 = nn.Linear(384, 192) 
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


        
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
    else:
        for iter_ in range(1, local_iter + 1):
            tau += 1
            # At every batch sample with replacement, in-batch sample w.o replacement
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
    correct = 0
    total = 0
    loss_ = 0 
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss(reduction='sum')
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_ += float(criterion(outputs, labels))
            _, predicted = torch.max(  # pylint: disable=no-member
                outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy, loss_/testloader.__len__()
