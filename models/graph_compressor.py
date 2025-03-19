import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Define the Graph Autoencoder
class GraphAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initializes the Graph Autoencoder model.

        Args:
            input_dim (int): The dimension of the input features.
            hidden_dim (int): The dimension of the hidden layer.
            output_dim (int): The dimension of the output layer.
        """
        super(GraphAutoencoder, self).__init__()
        self.encoder = GCNConv(input_dim, hidden_dim)
        self.decoder = GCNConv(hidden_dim, output_dim)

    def forward(self, data) -> torch.Tensor:
        """Performs a forward pass of the autoencoder.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices in COO format.

        Returns:
            torch.Tensor: The reconstructed node features.
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # Encode the input features
        x = F.relu(self.encoder(x, edge_index))
        # Decode to reconstruct the input
        x = self.decoder(x, edge_index)
        return x

# Example usage
if __name__ == "__main__":
    # Sample graph data
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    x = torch.rand((4, 3))  # Random input features for 4 nodes with feature dimension 3

    # Create the graph data object
    data = Data(x=x, edge_index=edge_index)

    # Initialize the model
    input_dim = 3
    hidden_dim = 2
    output_dim = 3
    model = GraphAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Forward pass
    reconstructed_x = model(data.x, data.edge_index)
    print("Reconstructed Node Features:\n", reconstructed_x)
