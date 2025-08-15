import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMGenerator(nn.Module):
    def __init__(self, static_dim, hidden_dims, num_blocks, channel_sizes):
        """
        Generates FiLM parameters (γ, β) for conditioning TCN blocks
        
        Args:
            static_dim: Dimension of static parameters (6 submarine properties)
            hidden_dims: List of hidden layer dimensions for MLP
            num_blocks: Number of TCN blocks to generate parameters for
            channel_sizes: List of channel sizes for each TCN block
        """
        super().__init__()
        # Validate inputs
        if len(channel_sizes) != num_blocks:
            raise ValueError("channel_sizes must have length equal to num_blocks")
        
        # Build shared MLP backbone
        layers = []
        in_dim = static_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden
        
        self.shared_mlp = nn.Sequential(*layers)
        
        # Create FiLM heads for each block
        self.film_heads = nn.ModuleList()
        for i in range(num_blocks):
            self.film_heads.append(
                nn.Linear(in_dim, 2 * channel_sizes[i])  # Output γ and β
            )
        
    def forward(self, theta):
        """
        Args:
            theta: Static parameters [batch_size, static_dim]
        
        Returns:
            film_params: List of (gamma, beta) tuples for each TCN block
        """
        # Process through shared MLP
        features = self.shared_mlp(theta)  # [batch_size, last_hidden_dim]
        
        # Generate FiLM parameters for each block
        film_params = []
        for head in self.film_heads:
            out = head(features)  # [batch_size, 2 * channels]
            gamma, beta = torch.chunk(out, 2, dim=-1)  # Split into γ and β
            film_params.append((gamma, beta))
            
        return film_params