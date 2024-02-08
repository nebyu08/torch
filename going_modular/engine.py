import torch
import torch.nn as nn
from typing import List,Tuple
from tqdm import tqdm

def train_step(
              model:nn.Module,
              train_data:torch.utils.data.DataLoader,
              loss_fn:torch.nn,
              optimizer:torch.optim.Optimizer,
              device:torch.device,
              )->Tuple[float,float]:
    
    train_loss,train_acc=0,0
    model.train()
    
    for batch,(x,y) in enumerate(train_data):
        x,y=x.to(device),y.to(device)
        pred_logits=model(x)
        
        preds=torch.argmax(torch.softmax(pred_logits,dim=1),dim=1)
        loss=loss_fn(pred_logits,y)
        
        train_loss+=loss.item()
        train_acc+=(preds==y).sum().item()/len(train_data)
    
        #lets zero grad the gradient
        optimizer.zero_grad()
        #lets backpropagate
        loss.backward()
        #lets update the parameter
        optimizer.step()
        
    train_loss/=len(train_data)
    train_acc/=len(train_data)
    
    return train_loss,train_acc
    
def test_step(model:nn.Module,
             test_data:torch.utils.data.DataLoader,
             loss_fn:torch.nn,
             device:torch.device="cpu")->Tuple[float,float]:
    
    test_loss,test_acc=0,0
    model.eval()
    with torch.inference_mode():
        for batch,(x,y) in enumerate(test_data):
        
            x,y=x.to(device),y.to(device)
            y_pred_logits=model(x)
            
            y_pred=torch.argmax(torch.softmax(y_pred_logits,dim=1),dim=1)
            loss=loss_fn(y_pred_logits,y).item()
            
            test_loss+=loss
            test_acc=(y==y_pred).sum().item()/len(test_data)
    test_loss/=len(test_data)
    test_acc/=len(test_data)
    return test_loss,test_acc

def train_model(model:nn.Module,
               train_data:torch.utils.data.DataLoader,
               test_data:torch.utils.data.DataLoader,
               loss_fn:torch.nn,
                optimizer:torch.optim.Optimizer,
                epochs:int,
               device:torch.device):
    #TRAIN MDOEL
    results={"train_loss":[],
            "train_acc":[],
            "test_loss":[],
            "test_acc":[]}
    
    train_loss,train_acc=0,0
    for epoch in tqdm(range(epochs)):
        train_loss,train_acc=train_step(model=model,
                                       train_data=train_data,
                                       loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device
                                       )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        
        #TEST THE MDODEL
        test_loss,test_acc=0,0
        test_loss,test_acc=test_step(model=model,
                             test_data=test_data,
                            loss_fn=loss_fn,
                            device=device
                            )
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results
