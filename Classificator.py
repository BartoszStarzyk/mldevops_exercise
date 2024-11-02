import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Tuple

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
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


class FashionMNISTNet(nn.Module):
    """
    A neural network model for classifying Fashion MNIST images.

    Attributes:
        fc1 (nn.Linear): Fully connected layer from input to hidden layer.
        fc2 (nn.Linear): Fully connected hidden layer.
        fc3 (nn.Linear): Fully connected layer from hidden to output layer.
        relu (nn.ReLU): ReLU activation function.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self) -> None:
        """Initializes the layers of the neural network."""
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)  # First fully connected layer (input to hidden)
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer (hidden to hidden)
        self.fc3 = nn.Linear(256, 10)  # Third fully connected layer (hidden to output)
        self.relu = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout(0.2)  # Dropout for regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor, a batch of flattened 28x28 images.

        Returns:
            torch.Tensor: The output logits for each class.
        """
        x = x.view(-1, 28 * 28)  # Flatten the input (28x28 image) to a vector
        x = self.relu(self.fc1(x))  # Apply ReLU to first layer output
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc2(x))  # Apply ReLU to second layer output
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # Output layer (logits)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FashionMNISTNet().to(device)  # Move the model to the GPU if available
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

print("Training")


def train_model(
    model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int = 5
) -> None:
    """
    Trains the neural network model on the training dataset.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for model parameters.
        num_epochs (int): Number of training epochs. Defaults to 5.
    """
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


# Train the model for 5 epochs
train_model(model, train_loader, criterion, optimizer, num_epochs=5)


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> None:
    """
    Evaluates the trained model on the test dataset and prints accuracy.

    Args:
        model (nn.Module): Trained neural network model.
        test_loader (DataLoader): DataLoader for the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class (index of the max logit)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


# Evaluate the trained model
evaluate_model(model, test_loader)
