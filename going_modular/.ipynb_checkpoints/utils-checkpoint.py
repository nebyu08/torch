import torch
import torch.nn as nn
from pathlin import Path

def save_model(model:nn.Module,
              target:str,
              model_name:str):
    
    target_path=Path(target)
    target_path.mkdir(parents_ok=True,exists=True)
    
    torch.save(model,target_path)
    #assert the externsion of the model
    
    assert model_name.endswith(".pt") or model_name.endswith(".pth"),"model extension should be .pt or .pth"
    model_path=target_path/model_name  #created the directory

    print(f"saving the model into {model_path}")
    torch.save(model.state_dict(),f=model_path)
