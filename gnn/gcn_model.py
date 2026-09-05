import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class NutriTwinGCN(torch.nn.Module):

    def __init__(
        self,
        input_features,
        hidden_features=32,
        output_features=16
    ):

        super().__init__()

        self.conv1 = GCNConv(
            input_features,
            hidden_features
        )

        self.conv2 = GCNConv(
            hidden_features,
            output_features
        )

    def forward(
        self,
        x,
        edge_index
    ):

        # First graph convolution
        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        # Second graph convolution
        x = self.conv2(
            x,
            edge_index
        )

        return x


if __name__ == "__main__":

    print("=" * 60)
    print("NutriTwinNet Graph Convolutional Network")
    print("=" * 60)

    model = NutriTwinGCN(
        input_features=5,
        hidden_features=32,
        output_features=16
    )

    print(model)

    print("\nGNN model created successfully.")