import torch
from torch.utils.data import DataLoader

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

    accuracy = 100 * correct / total

    return accuracy


if __name__ == "__main__":
    # Test the evaluation function
    _, test_data = load_data()

    model = FashionCNN()

    accuracy = evaluate_model(model, test_data)

    print(f"Untrained model accuracy: {accuracy:.2f}%")