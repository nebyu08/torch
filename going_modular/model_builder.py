import torch
import torch.nn as nn

class TinnyVGG(nn.Module):
    """ makes tunny vgg model """
    def __init__(self,input_shape:int,
                 n_hidden:int,
                output_shape:int):
        super().__init__()
        
        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                             out_channels=n_hidden,
                             kernel_size=3,
                             stride=1,
                             padding=0),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=n_hidden,
                     out_channels=n_hidden,
                     kernel_size=3,
                     stride=1,
                     padding=0),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2,
                        stride=2,
                        padding=0)
        )
        self.block2=nn.Sequential(
            nn.Conv2d(in_channels=n_hidden,
                      out_channels=n_hidden,
                     kernel_size=3,
                     stride=1,
                     padding=0),
            
             nn.ReLU(),
            
             nn.Conv2d(in_channels=n_hidden,
                     out_channels=n_hidden,
                     kernel_size=3,
                     stride=1,
                     padding=0),
            nn.ReLU(),
           
            nn.MaxPool2d(kernel_size=2,
                      stride=2,
                      padding=0)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_hidden*13*13,
                      out_features=output_shape)
        )
    def forward(self,x:torch.Tensor):
        #print(f"initial shape: {x.shape}")
        x=self.block1(x)
        #print(f"after block_1: {x.shape}")
        x=self.block2(x)
        #print(f"after block_2: {x.shape}")
        x=self.classifier(x)
        #print(f"final shape:{x.shape} ")
        return x
