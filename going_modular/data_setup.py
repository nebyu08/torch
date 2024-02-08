"""makes training and testing dataloader"""
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def create_dataloader(train_dir:str,
                     test_dir:str,
                    transform:transforms.Compose,
                     batch_size:int
                     ):
    
    """takes in filepath and returns train data loader,test data loader and numbers of classes"""
 
    train_dataset=ImageFolder(
        root=train_dir,
        transform=transform,
        )
    
    test_dataset=ImageFolder(
        root=test_dir,
        transform=transform
    )

    class_names=train_dataset.classes
    
    train_dataloader=DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )
    
    test_dataloader=DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_dataloader,test_dataloader,class_names
