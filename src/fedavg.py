import copy
import random

import numpy as np
import torch

from data import load_data
from model import FashionCNN
from partition import (
    partition_iid,
    partition_non_iid,
    partition_test_like_train
)
from client import train_client
from evaluate import evaluate_model, evaluate_clients


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


def train_fedavg(
    rounds=2,
    num_clients=10,
    distribution="non_iid",
    alpha=0.5,
    seed=42
):
    # Set random seeds for reproducible results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load FashionMNIST dataset
    train_data, test_data = load_data()

    # Partition training data across clients
    if distribution == "iid":
        clients = partition_iid(
            train_data,
            num_clients=num_clients,
            seed=seed
        )

    elif distribution == "non_iid":
        clients = partition_non_iid(
            train_data,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed
        )

    else:
        raise ValueError(
            "Distribution must be 'iid' or 'non_iid'"
        )

    # Initialize global model
    global_model = FashionCNN()

    print(f"Distribution: {distribution}")
    print(f"Number of clients: {num_clients}")
    print(f"Training rounds: {rounds}")

    if distribution == "non_iid":
        print(f"Dirichlet alpha: {alpha}")

    # Federated training rounds
    for round_num in range(rounds):
        print(
            f"\n=== Round {round_num + 1}/{rounds} ===",
            flush=True
        )

        client_weights = []
        client_sizes = []
        client_losses = []

        # Train each client independently
        for client_id, indices in enumerate(clients):
            print(
                f"Training client {client_id} "
                f"({len(indices)} images)...",
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

        # Aggregate client models using FedAvg
        new_weights = federated_average(
            client_weights,
            client_sizes
        )

        global_model.load_state_dict(new_weights)

        # Calculate weighted average training loss
        avg_loss = sum(
            loss * size
            for loss, size in zip(client_losses, client_sizes)
        ) / sum(client_sizes)

        # Evaluate global model on full test dataset
        accuracy = evaluate_model(
            global_model,
            test_data
        )

        print(
            f"Round {round_num + 1} average loss: "
            f"{avg_loss:.4f}",
            flush=True
        )

        print(
            f"Round {round_num + 1} test accuracy: "
            f"{accuracy:.2f}%",
            flush=True
        )

    # Create held-out test partitions that approximately
    # match each client's training class distribution
    client_test_partitions = partition_test_like_train(
        train_data,
        test_data,
        clients,
        seed=seed
    )

    # Evaluate fairness across clients
    print("\n=== Per-Client Fairness Evaluation ===")

    client_accuracies = evaluate_clients(
        global_model,
        test_data,
        client_test_partitions
    )

    print("\nFedAvg training complete!")

    return global_model


if __name__ == "__main__":
    model = train_fedavg(
        rounds=2,
        num_clients=10,
        distribution="non_iid",
        alpha=0.5,
        seed=42
    )