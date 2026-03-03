import torch
import numpy as np


def train_autoencoder(model, windows, epochs=20, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    dataset = torch.tensor(windows, dtype=torch.float32).to(device)

    for epoch in range(epochs):

        optimizer.zero_grad()

        output = model(dataset)

        loss = criterion(output, dataset)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    return model