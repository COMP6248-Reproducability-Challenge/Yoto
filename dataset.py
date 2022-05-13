import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split
import os
from torchvision.datasets import ImageFolder

CIFAR_data_dir = './data/cifar10'
SHAPES_data_dir = ''

def prepare_CIFAR_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(32)])

    batch_size = 128

    trainset = ImageFolder(CIFAR_data_dir + '/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = ImageFolder(CIFAR_data_dir + '/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def prepare_SHAPES_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(32)])

    batch_size = 128
    
    images_train = ImageFolder('data\shapes3dtrain', transform=transform)
    images_test = ImageFolder('data\shapes3dtest', transform=transform)

    trainloader = torch.utils.data.DataLoader(images_train, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(images_test, batch_size=batch_size, shuffle=True, num_workers=0)

    return trainloader, testloader