
data_dir = './data'

import torch

import torchvision.transforms as transforms
import torchvision
import pytorch_lightning as pl


class ValTrainHelper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)



class CIFAR10(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_val()
    def train_val(self):
        train_val = torchvision.datasets.CIFAR10(root = './data', train = True, download = True)
        self.train, self.val = torch.utils.data.random_split(train_val, [int(0.95 * len(train_val)), len(train_val) - int(0.95 * len(train_val))])

    def train_dataloader(self):
        trainset = ValTrainHelper(self.train,self.transform_train)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True, num_workers = 2)

        return trainloader


    def val_dataloader(self):
        valset = ValTrainHelper(self.val, self.transform_test)

        valloader = torch.utils.data.DataLoader(valset, batch_size = 64, shuffle = False, num_workers = 2)

        return valloader

    def test_dataloader(self):

        testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True,
                                               transform = self.transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = False, num_workers = 2)
        return testloader


