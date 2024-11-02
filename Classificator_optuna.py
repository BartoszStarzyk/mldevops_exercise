"""sn_zadanie_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19u8zfjXQ7CURVwzvS2PQP2ru5lTspcqV
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import optuna

# Define transformations for the training and test datasets
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize images to [-1, 1] range
    ]
)

# Download and load the Fashion MNIST training and test datasets
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Define data loaders for training and test datasets
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


class FashionMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)  # First fully connected layer (input to hidden)
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer (hidden to hidden)
        self.fc3 = nn.Linear(256, 10)  # Third fully connected layer (hidden to output)
        self.relu = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout(0.2)  # Dropout for regularization

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input (28x28 image) to a vector
        x = self.relu(self.fc1(x))  # Apply ReLU to first layer output
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc2(x))  # Apply ReLU to second layer output
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # Output layer (logits)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    return running_loss / len(train_loader)


# Train the model for 5 epochs
def objective(trial):
    lr = trial.suggest_float("lr", 0.0005, 0.002)
    model = FashionMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = train_model(model, train_loader, criterion, optimizer, num_epochs=5)
    return loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)
print(study.best_params)
