import torch 
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,1)
    
    def forward(self,x):
        out=self.fc1(x)
        return out

model=SimpleNN()

criterion=nn.MSELoss() #regression loss

optimizer=optim.Adam(model.parameters(),lr=0.1)

#lets train a 3x+1

x_train=torch.tensor([[2.0],[3.0],[4.0],[5.0]])
y_train=3*x_train +1

#Training Loop 

epochs=1000

for epoch in range(epochs):
    model.train()

    #forward pass
    predictions=model(x_train)

    # compute loss

    loss=criterion(predictions, y_train)

    #backpropagation

    optimizer.zero_grad() # erase old gradient info

    loss.backward()   # compute new gradients

    optimizer.step()  # update weights

    
    # check every 10 epochs

    if epoch % 10==0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")


# Test the model:

model.eval()

with torch.no_grad():
    value=torch.tensor([[8.0],[4.3]])
    prediction=model(value)
    for i in range(len(value)):
        print(f"Value: {value[i]} , prediction:{prediction[i].item():.2f}")
