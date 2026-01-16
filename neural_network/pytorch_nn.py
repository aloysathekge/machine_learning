import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,32)
        self.fc2=nn.Linear(32,8)
        self.fc3=nn.Linear(8,1)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        out=self.relu(self.fc1(x))
        out=self.relu(self.fc2(out))
        out=self.fc3(out)

        return out

model=SimpleNN()

criterion=nn.MSELoss() #regression loss

optimizer=optim.Adam(model.parameters(),lr=0.001)

# More training data for better learning
x_train=torch.linspace(0,10,100).reshape(-1,1)
y_train=x_train **2

dataset=TensorDataset(x_train,y_train)
train_loader=DataLoader(dataset,batch_size=2,shuffle=True)

def training_loop(model,optimizer,dataloader, loss_fn):
    model.train()

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()

        predictions=model(batch_x)

        loss=loss_fn(predictions,batch_y)

        loss.backward()

        optimizer.step()
    return loss

epochs=2500
total_loss=0
for epoch in range(epochs):
  loss= training_loop(model, optimizer,train_loader,criterion)
  total_loss += loss.item()
  if epoch % 200==0:
    print(f"Epoch : {epoch}, Loss:{loss.item():.4f}")

print(f"Average  Loss {total_loss/epochs:.4f}") 


# Test the model:
model.eval()

with torch.no_grad():
    print("\nTraining data predictions:")
    train_pred = model(x_train)
    for i in range(len(x_train)):
        print(f"x={x_train[i].item():.1f}, predicted={train_pred[i].item():.1f}, actual={y_train[i].item():.1f}")
   