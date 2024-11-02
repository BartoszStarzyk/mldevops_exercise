import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Tuple
import wandb

# Log in to wandb
wandb.login()


class Autoencoder(nn.Module):
    """
    A simple Autoencoder model with a symmetric encoder-decoder architecture.

    Attributes:
        encoder (nn.Sequential): Neural network layers for encoding input data.
        decoder (nn.Sequential): Neural network layers for decoding encoded data.
    """

    def __init__(self) -> None:
        """Initializes the Autoencoder model with predefined encoder and decoder architectures."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Autoencoder.

        Args:
            x (torch.Tensor): Input tensor, flattened image.

        Returns:
            torch.Tensor: Reconstructed output tensor, same shape as input.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def visualize_reconstruction(model: nn.Module, data_loader: DataLoader, num_images: int = 10) -> None:
    """
    Visualizes original and reconstructed images from the Autoencoder.

    Args:
        model (nn.Module): Trained Autoencoder model.
        data_loader (DataLoader): DataLoader containing images for reconstruction.
        num_images (int): Number of images to visualize. Defaults to 10.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for data in data_loader:
            img, _ = data
            img = img.view(img.size(0), -1)  # Flatten the image
            recon = model(img)  # Reconstruct the image
            img = img.view(-1, 28, 28)  # Reshape for visualization
            recon = recon.view(-1, 28, 28)  # Reshape reconstructed images

            for i in range(num_images):
                fig, axes = plt.subplots(1, 2, figsize=(6, 3))

                # Display original image
                axes[0].imshow(img[i], cmap="gray")
                axes[0].set_title("Original")
                axes[0].axis("off")

                # Display reconstructed image
                axes[1].imshow(recon[i], cmap="gray")
                axes[1].set_title("Reconstructed")
                axes[1].axis("off")

                plt.show()
            break  # Only display one batch of images


# Load the Fashion MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_dataset: Dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)

# Split into training and validation sets
train_size: int = int(0.8 * len(full_dataset))
val_size: int = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

n_epochs: int = 5
lr: float = 0.001

# Initialize the model, loss function, and optimizer
model = Autoencoder()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=lr)

# Initialize wandb run
run = wandb.init(
    project="SN-Autoencoder",
    config={
        "learning_rate": lr,
        "epochs": n_epochs,
    },
)

# Log the code file as an artifact
artifact = wandb.Artifact("autoencoder_script", type="code")
artifact.add_file("Autoencoder.py")
wandb.log_artifact(artifact)

# Training the Autoencoder with validation loss calculation
for epoch in range(n_epochs):
    model.train()  # Set the model to training mode
    train_loss: float = 0.0

    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)  # Flatten the image

        # Forward pass
        output = model(img)
        loss = criterion(output, img)  # MSE between input and output

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Calculate average training loss for the epoch
    avg_train_loss: float = train_loss / len(train_loader)

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss: float = 0.0

    with torch.no_grad():
        for data in val_loader:
            img, _ = data
            img = img.view(img.size(0), -1)  # Flatten the image
            output = model(img)
            loss = criterion(output, img)
            val_loss += loss.item()

    # Calculate average validation loss for the epoch
    avg_val_loss: float = val_loss / len(val_loader)

    # Log losses to wandb
    print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    wandb.log({"training_loss": avg_train_loss, "validation_loss": avg_val_loss})

# Visualize the results
visualize_reconstruction(model, val_loader)

# Finish the wandb run
wandb.finish()
