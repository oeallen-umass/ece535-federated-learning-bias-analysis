
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data import load_data
from model import FashionCNN
from partition import partition_iid


def train_client(global_model, dataset, indices,
                 epochs=1, batch_size=64, lr=0.01):

    # Create a separate copy of the global model
    local_model = copy.deepcopy(global_model)
    local_model.train()

    # Give the client only its assigned images
    client_data = Subset(dataset, indices)

    loader = DataLoader(
        client_data,
        batch_size=batch_size,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=lr
    )

    total_loss = 0
    total_samples = 0

    for epoch in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad()

            predictions = local_model(images)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    average_loss = total_loss / total_samples

    return local_model.state_dict(), average_loss


if __name__ == "__main__":
    # Load the clothing dataset
    train_data, _ = load_data()

    # Divide it among 10 simulated clients
    clients = partition_iid(train_data)

    # Create starting AI model
    global_model = FashionCNN()

    print("Training Client 0...")

    # Test training on just one client
    updated_weights, loss = train_client(
        global_model,
        train_data,
        clients[0]
    )

    print("Training complete!")
    print(f"Average training loss: {loss:.4f}")
    print("Model parameters updated:", len(updated_weights))