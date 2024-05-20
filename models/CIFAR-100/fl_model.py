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
    """Generator for CIFAR-100 dataset."""

    # Extract CIFAR-100 data using torchvision datasets
    def read(self, path, testset_device = 'cpu'):
        transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.trainset = datasets.CIFAR100(
            path, train=True, download=True, transform=transf)
        self.testset = datasets.CIFAR100(
            path, train=False, transform=transf)
        self.labels = list(self.trainset.classes)
        if testset_device == 'cuda':
            x = torch.Tensor(self.testset.data/255).permute(0,3,1,2)
            self.testset.data = transf.transforms[-1](x)
            self.testset = TensorDataset(torch.tensor(self.testset.data).to(testset_device), torch.tensor(self.testset.targets).to(testset_device))
    
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo




class Net(nn.Module): # ResNet-18 by FedDyn
    def __init__(self):
        super(Net,self).__init__()
        resnet18 = models.resnet18()
        resnet18.fc = nn.Linear(512, 100)

        # Change BN to GN 
        resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

        resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

        resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

        assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
        self.model = resnet18
    def forward(self, x):
        return self.model(x)


########
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2,planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2,planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2,self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class Net(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=100):
        super(Net, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2,64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out






        
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
                inputs = model.transform_train(inputs)
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
            inputs = model.transform_train(inputs)
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
