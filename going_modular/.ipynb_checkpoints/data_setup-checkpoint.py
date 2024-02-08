"""makes training and testing dataloader"""
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def create_dataloader(train_dir:str,
                     test_dir:str,
                     batch_numbers:int
                     ):
    """takes in filepath and returns train data loader,test data loader and numbers of classes"""
    #transforms
    train_transform=transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    
    test_transform=transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor()
    ])
    train_dataset=ImageFolder(
    root=train_dir,
    transform=train_transform,
    )
    test_dataset=ImageFolder(
        root=test_dir,
        transform=test_transform
    )
