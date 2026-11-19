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
from results import save_round_results, save_client_results


def afl_average(client_weights, mixture_weights):
    """
    Combine local models using AFL's adaptive client weights.
    """
    averaged_weights = copy.deepcopy(client_weights[0])

    for key in averaged_weights:
        averaged_weights[key] = sum(
            state[key] * float(weight)
            for state, weight in zip(client_weights, mixture_weights)
        )

    return averaged_weights


def update_mixture_weights(
    mixture_weights,
    client_losses,
    step_size=0.1
):
    """
    Increase weights for clients with higher losses.
    Uses exponentiated-gradient updates.
    """
    losses = np.asarray(client_losses, dtype=float)

    # Stabilize the exponential calculation
    scores = step_size * losses
    scores = scores - np.max(scores)

    new_weights = mixture_weights * np.exp(scores)
    new_weights = new_weights / new_weights.sum()

    return new_weights


def train_afl(
    rounds=2,
    num_clients=10,
    distribution="non_iid",
    alpha=0.5,
    mixture_lr=0.1,
    seed=42
):
    # Reproducible experiments
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    train_data, test_data = load_data()

    # Partition training data
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

    # Initialize shared model
    global_model = FashionCNN()

    # AFL starts with equal client weights
    mixture_weights = np.ones(num_clients) / num_clients

    round_results = []

    print("Algorithm: AFL (adaptive client weighting)")
    print(f"Distribution: {distribution}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {rounds}")
    print(f"Mixture learning rate: {mixture_lr}")

    # Federated training
    for round_num in range(rounds):
        print(
            f"\n=== Round {round_num + 1}/{rounds} ===",
            flush=True
        )

        client_weights = []
        client_losses = []
        client_sizes = []

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
            client_losses.append(loss)
            client_sizes.append(len(indices))

        # Update persistent AFL mixture weights
        mixture_weights = update_mixture_weights(
            mixture_weights,
            client_losses,
            step_size=mixture_lr
        )

        # Combine models using updated mixture weights
        new_weights = afl_average(
            client_weights,
            mixture_weights
        )

        global_model.load_state_dict(new_weights)

        # Weighted average training loss for reporting
        avg_loss = sum(
            loss * size
            for loss, size in zip(client_losses, client_sizes)
        ) / sum(client_sizes)

        # Evaluate shared model
        accuracy = evaluate_model(
            global_model,
            test_data
        )

        print(
            f"Round {round_num + 1} average loss: "
            f"{avg_loss:.4f}"
        )

        print(
            f"Round {round_num + 1} test accuracy: "
            f"{accuracy:.2f}%"
        )

        print(
            "AFL mixture weights:",
            np.round(mixture_weights, 3).tolist()
        )

        round_results.append((
            round_num + 1,
            avg_loss,
            accuracy
        ))

    # Held-out client-specific test distributions
    client_test_partitions = partition_test_like_train(
        train_data,
        test_data,
        clients,
        seed=seed
    )

    # Evaluate fairness
    print("\n=== Per-Client Fairness Evaluation ===")

    client_accuracies = evaluate_clients(
        global_model,
        test_data,
        client_test_partitions
    )

    # Save results
    save_round_results(
        "afl",
        distribution,
        round_results
    )

    save_client_results(
        "afl",
        distribution,
        client_accuracies
    )

    print("\nAFL training complete!")

    return global_model


if __name__ == "__main__":
    model = train_afl(
        rounds=2,
        num_clients=10,
        distribution="non_iid",
        alpha=0.5,
        mixture_lr=0.1,
        seed=42
    )