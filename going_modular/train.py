

from data_setup import create_dataloader
from engine import train_model
from model_builder import TinnyVGG
from utils import save_model
from timeit import default_timer as timer
import torch.nn as nn
from torchvision import transforms
import torch

torch.manual_seed(42)
torch.cuda.manual_seed(42)


batch_size=32
learning_rate=0.01
n_neurons=10

train_dir="data/pizza_steak_sushi/train"
test_dir="data/pizza_steak_sushi/test"

data_transform=transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

device="cuda" if torch.cuda.is_available() else "cpu"


#lets define the data loader-->data sets
train_dataloader,test_dataloader,class_names=create_dataloader(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=batch_size
)

#make the model
model_v1=TinnyVGG(
    input_shape=3,
    output_shape=len(class_names),
    n_hidden=n_neurons
)

#lets set up the parameters
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(params=model_v1.parameters(),lr=learning_rate)

start_time=timer()
#lets train our model
epochs=5
results=train_model(
    model=model_v1,
    train_data=train_dataloader,
    test_data=test_dataloader,
    loss_fn=loss,
    optimizer=optimizer,
    epochs=epochs,
    device=device
)
end_time=timer()
print(f"total amount of time taken to train the model is: {end_time-start_time}")
print(results)

#lets save the model
save_model(model_v1,target="models",model_name="tinnyVGG_model.pt")
