
import copy

import torch

from data import load_data
from model import FashionCNN
from partition import partition_iid
from client import train_client
from evaluate import evaluate_model


def federated_average(client_weights, client_sizes):
    """Combine client models using dataset-size weighting."""
    total_samples = sum(client_sizes)

    averaged_weights = copy.deepcopy(client_weights[0])

    for key in averaged_weights:
        averaged_weights[key] = sum(
            weights[key] * (size / total_samples)
            for weights, size in zip(client_weights, client_sizes)
        )

    return averaged_weights


def train_fedavg(rounds=2, num_clients=10):
    # Load training and testing datasets
    train_data, test_data = load_data()

    # Divide training data among clients
    clients = partition_iid(
        train_data,
        num_clients=num_clients
    )

    # Create the global model
    global_model = FashionCNN()

    # Run federated learning rounds
    for round_num in range(rounds):
        print(
            f"\n=== Round {round_num + 1}/{rounds} ===",
            flush=True
        )

        client_weights = []
        client_sizes = []
        client_losses = []

        # Train each client separately
        for client_id, indices in enumerate(clients):
            print(
                f"Training client {client_id}...",
                flush=True
            )

            weights, loss = train_client(
                global_model,
                train_data,
                indices
            )

            client_weights.append(weights)
            client_sizes.append(len(indices))
            client_losses.append(loss)

        # Combine all client models using FedAvg
        new_weights = federated_average(
            client_weights,
            client_sizes
        )

        # Update the global model
        global_model.load_state_dict(new_weights)

        # Calculate average training loss
        avg_loss = sum(
            loss * size
            for loss, size in zip(client_losses, client_sizes)
        ) / sum(client_sizes)

        print(
            f"Round {round_num + 1} average loss: {avg_loss:.4f}",
            flush=True
        )

        # Evaluate global model on unseen test images
        accuracy = evaluate_model(
            global_model,
            test_data
        )

        print(
            f"Round {round_num + 1} test accuracy: {accuracy:.2f}%",
            flush=True
        )

    print("\nFedAvg training complete!")

    return global_model


if __name__ == "__main__":
    model = train_fedavg()