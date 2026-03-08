import torch
import torch.nn as nn
import torch.optim as optim


def train_autoencoder(model, train_loader, num_epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return losses


def train_lstm_autoencoder(model, train_loader, num_epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            # batch: (batch_size, seq_len, input_dim) — already 3D from LSTMWindowDataset
            outputs = model(batch)

            # Both outputs and batch are (batch_size, seq_len, input_dim)
            loss = criterion(outputs, batch)

            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent exploding gradients common in LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return losses
