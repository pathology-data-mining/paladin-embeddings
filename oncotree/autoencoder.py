import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

def train_autoencoder(model, train_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            data = batch[0].to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return losses

def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('oncotree/autoencoder_loss.png')
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load embeddings
    with open('ncit/oncotree_embeddings.json', 'r') as f:
        embeddings_dict = json.load(f)
    
    # Convert to numpy array
    embeddings = np.array(list(embeddings_dict.values()))
    codes = list(embeddings_dict.keys())
    
    # Convert to PyTorch tensor
    embeddings_tensor = torch.FloatTensor(embeddings)
    
    # Create data loader
    dataset = TensorDataset(embeddings_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model parameters
    input_dim = embeddings.shape[1]  # 224
    latent_dim = 32  # Compressed dimension
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim, latent_dim).to(device)
    
    # Train model
    num_epochs = 100
    learning_rate = 0.001
    losses = train_autoencoder(model, train_loader, num_epochs, learning_rate, device)
    
    # Plot training loss
    plot_loss(losses)
    
    # Generate compressed embeddings
    model.eval()
    with torch.no_grad():
        compressed_embeddings = model.encode(embeddings_tensor).cpu().numpy()
    
    # Save compressed embeddings
    compressed_dict = {code: emb.tolist() for code, emb in zip(codes, compressed_embeddings)}
    with open('oncotree/compressed_embeddings.json', 'w') as f:
        json.dump(compressed_dict, f)
    
    print(f"Original embedding dimension: {input_dim}")
    print(f"Compressed embedding dimension: {latent_dim}")
    print(f"Compression ratio: {input_dim/latent_dim:.2f}x")

if __name__ == "__main__":
    main() 