import torch.nn as nn

class LSTMAutoencoder(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=32, latent_dim=16):
        super().__init__()

        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )

        self.latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            hidden_dim,
            input_dim,
            batch_first=True
        )

    def forward(self, x):

        # x shape: (batch, seq_len, input_dim)

        _, (h, _) = self.encoder(x)

        z = self.latent(h[-1])

        dec_input = self.decoder_input(z).unsqueeze(1)
        dec_input = dec_input.repeat(1, x.size(1), 1)

        out, _ = self.decoder(dec_input)

        return out