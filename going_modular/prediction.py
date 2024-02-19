import torch
import torchvision
from torchvision import transforms
from PIL import Image
from typing import Tuple,List
import matplotlib.pyplot as plt
import torch.nn as nn

def custom_predictions(
    model:nn.Module,
    img_dir:str,
    class_label:List[str],
    img_size:Tuple[int,int]=(224,224),
    device:torch.device="cpu",
    image_transform:torchvision.transforms=None,
  ):
    
    
    """ this function is used for displaying the image and the label predicted by the model"""
    #open image

    img=Image.open(img_dir)


    if image_transform is not None:
      transorm=image_transform

    else:

      #make the transform manually
      transform=transforms.Compose([
          transforms.ToTensor(),
          transforms.Resize(img_size),
          transforms.Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]
          )
      ])
    

    model.to(device)
    model.eval()
    with torch.inference_mode():
      transformed_image=transform(img)
      y_pred_logits=model(transformed_image.unsqueeze(dim=0))
      y_pred=torch.argmax(torch.softmax(y_pred_logits,dim=1),dim=1)
      label_pred=class_names[y_pred]

    plt.figure()
    plt.imshow(img)
    plt.title(f"pred {label_pred} | prob {y_pred_logits.max():.2f}")
    plt.axis("OFF")