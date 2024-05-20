import load_data
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import TensorDataset
# Training settings
log_interval = 10

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device(  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')


class Generator(load_data.Generator):
    """Generator for MNIST dataset."""

    # Extract MNIST data using torchvision datasets
    def read(self, path, testset_device = 'cpu'):
        transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
        self.trainset = datasets.MNIST(
            path, train=True, download=True, transform=transf)
        self.testset = datasets.MNIST(
            path, train=False, transform=transf)
        self.labels = list(self.trainset.classes)
        if testset_device == 'cuda':
            x = torch.Tensor(self.testset.data/255).unsqueeze(1)
            self.testset.data = transf.transforms[-1](x)
            self.testset = TensorDataset(torch.tensor(self.testset.data).to(testset_device), torch.tensor(self.testset.targets).to(testset_device))

from torch.nn import Module

#FedDyn - MNIST model
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_cls = 10
        self.fc1 = nn.Linear(1 * 28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, self.n_cls)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
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
    test_loss = 0
    correct = 0
    total = len(testloader.dataset)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for image, label in testloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            # sum up batch loss
            test_loss += criterion(output, label).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / total

    return accuracy, test_loss/testloader.__len__()
