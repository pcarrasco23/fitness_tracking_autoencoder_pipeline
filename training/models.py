import torch
import torch.nn as nn


# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=8):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoding_dim, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (hidden, cell) = self.lstm(x)
        # Use final layer's hidden state as sequence summary
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        encoded = torch.tanh(self.fc(last_hidden))  # (batch, encoding_dim)
        return encoded


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoding_dim, seq_len, num_layers, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc = nn.Linear(encoding_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, encoded):
        # encoded: (batch, encoding_dim)
        # Project encoding and use as input at every timestep
        decoder_input = torch.relu(self.fc(encoded))  # (batch, hidden_dim)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)

        output, _ = self.lstm(decoder_input)  # (batch, seq_len, hidden_dim)
        reconstruction = self.output_fc(output)  # (batch, seq_len, input_dim)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """
    Proper LSTM Autoencoder for time series.

    Input shape:  (batch, seq_len, input_dim)
    Output shape: (batch, seq_len, input_dim)

    Args:
        input_dim:    Number of features per timestep
        seq_len:      Number of timesteps in each window
        hidden_dim:   LSTM hidden state size
        encoding_dim: Bottleneck latent dimension
        num_layers:   Number of stacked LSTM layers
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int = 64,
        encoding_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, encoding_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(input_dim, hidden_dim, encoding_dim, seq_len, num_layers, dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def encode(self, x):
        """Get latent representation for downstream tasks."""
        return self.encoder(x)
