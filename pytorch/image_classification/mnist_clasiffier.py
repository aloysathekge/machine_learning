import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data  import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()

        # flatten the Image
        self.flatten=nn.Flatten()

        # Forward network
        self.fc1=nn.Linear(28*28,128)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(128,64)
        self.relu2=nn.ReLU()
        self.fc3=nn.Linear(64,10)

    def forward(self,x):
        x=self.flatten(x)    
        x=self.relu1(self.fc1(x))
        x=self.relu2(self.fc2(x))
        x=self.fc3(x)
        return x

# PREPARE DATA

def getData_Loaders(batch_size=64):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307),(0.3081))
    ])

    # Download and load training data
    train_dataset=datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform

    )

    test_dataset=datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create dataloaders

    train_loader=DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
    
    return train_loader, test_loader

# Training the model
def train(model, train_loader,optimizer,criterion, epoch):
    model.train()
    train_loss=0
    correct=0
    total=0

    for batch_idx, (data,target) in enumerate(train_loader):
        data,target=data.to(device), target.to(device)

        #zero the gradients
        optimizer.zero_grad()

        #forward pass
        output=model(data)

        # loss

        loss=criterion(output,target)

        # backward pass
        loss.backward()

        # update
        optimizer.step()

        # Track metrics
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # print progress 
        if batch_idx % 100==0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]'
                f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}'

            )
        
    avg_loss=train_loss/len(train_loader)
    accuracy=100. * correct/total

    print(f"\nTrain set: Average loss:{avg_loss:.4f},"
            f'Accuracy : {correct}/{total} ({accuracy:.2f}%)\n'
    )

    return avg_loss,accuracy

def test(model,test_loader,criterion):

    model.eval()
    test_loss=0
    correct=0
    total=0

    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device), target.to(device)

            output=model(data)

            test_loss+=criterion(output,target).item()

            _,predicted=output.max(1)

            total +=target.size(0)

            correct +=predicted.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Test set: Average loss: {avg_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    
    return avg_loss, accuracy


# Main training loop
def main():

    #hyperparameters
    batch_size=64
    learning_rate=0.001
    epochs=5

    print("Loading Mnist dataset..")
    train_loader,test_loader=getData_Loaders(batch_size)

    # Init model

    model=ImageModel().to(device)
    print(f"\nModel Architecture:\n{model}\n")

    # Loss and Optimizer

    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    criterion=nn.CrossEntropyLoss()

    
    # Trianing loop
    train_losses,train_accs=[],[]
    test_losses, test_accs = [], []

    for epoch in range(1, epochs+1):
        print(f"{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*50}")

        train_loss, train_acc =train(model,train_loader, optimizer, criterion,epoch )
        test_loss, test_acc=test(model,test_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)


        #Save the Model

    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved as mnist_model.pth")

    #plot results

    # plot_results(train_losses, test_losses,train_accs,test_accs)


# 6. Visualization function
# def plot_results(train_losses, test_losses, train_accs, test_accs):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
#     # Plot losses
#     ax1.plot(train_losses, label='Train Loss', marker='o')
#     ax1.plot(test_losses, label='Test Loss', marker='o')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax1.set_title('Training and Test Loss')
#     ax1.legend()
#     ax1.grid(True)
    
#     # Plot accuracies
#     ax2.plot(train_accs, label='Train Accuracy', marker='o')
#     ax2.plot(test_accs, label='Test Accuracy', marker='o')
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Accuracy (%)')
#     ax2.set_title('Training and Test Accuracy')
#     ax2.legend()
#     ax2.grid(True)
    
#     plt.tight_layout()
#     plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
#     print("Training results saved as 'training_results.png'")
#     plt.close()


if __name__ == '__main__':
    main()








