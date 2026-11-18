import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import load_data
from model import FashionCNN


def evaluate_model(model, test_data, batch_size=128):
    model.eval()

    loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images)
            predicted_classes = predictions.argmax(dim=1)

            correct += (
                predicted_classes == labels
            ).sum().item()

            total += labels.size(0)

    return 100 * correct / total


def evaluate_clients(model, test_data, client_partitions):
    """
    Evaluate the global model separately for each client.
    Each client receives a held-out test subset.
    """
    accuracies = []

    for client_id, indices in enumerate(client_partitions):
        client_test_data = Subset(test_data, indices)

        accuracy = evaluate_model(model, client_test_data)
        accuracies.append(accuracy)

        print(
            f"Client {client_id} test accuracy: "
            f"{accuracy:.2f}%"
        )

    print("\nFairness metrics:")
    print(f"Mean client accuracy: {np.mean(accuracies):.2f}%")
    print(f"Worst client accuracy: {np.min(accuracies):.2f}%")
    print(f"Accuracy standard deviation: {np.std(accuracies):.2f}")

    return accuracies


if __name__ == "__main__":
    _, test_data = load_data()
    model = FashionCNN()

    accuracy = evaluate_model(model, test_data)

    print(f"Untrained model accuracy: {accuracy:.2f}%")