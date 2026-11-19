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


def term_average(client_weights, client_sizes, client_losses, tilt=1.0):
    """
    Aggregate client models with greater weight on high-loss clients.
    This is a simplified client-level TERM approximation.
    """
    sizes = np.asarray(client_sizes, dtype=float)
    losses = np.asarray(client_losses, dtype=float)

    # Start with FedAvg dataset-size weights
    base_weights = sizes / sizes.sum()

    # Exponential loss weighting with numerical stabilization
    scores = tilt * losses
    scores = scores - np.max(scores)

    weights = base_weights * np.exp(scores)
    weights = weights / weights.sum()

    averaged_weights = copy.deepcopy(client_weights[0])

    for key in averaged_weights:
        averaged_weights[key] = sum(
            state[key] * float(weight)
            for state, weight in zip(client_weights, weights)
        )

    return averaged_weights, weights


def train_term(
    rounds=2,
    num_clients=10,
    distribution="non_iid",
    alpha=0.5,
    tilt=1.0,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_data, test_data = load_data()

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
        raise ValueError("Distribution must be 'iid' or 'non_iid'")

    global_model = FashionCNN()
    round_results = []

    print("Algorithm: TERM (simplified)")
    print(f"Distribution: {distribution}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {rounds}")
    print(f"Tilt: {tilt}")

    for round_num in range(rounds):
        print(f"\n=== Round {round_num + 1}/{rounds} ===", flush=True)

        client_weights = []
        client_sizes = []
        client_losses = []

        for client_id, indices in enumerate(clients):
            print(
                f"Training client {client_id} ({len(indices)} images)...",
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

        new_weights, aggregation_weights = term_average(
            client_weights,
            client_sizes,
            client_losses,
            tilt=tilt
        )

        global_model.load_state_dict(new_weights)

        avg_loss = sum(
            loss * size
            for loss, size in zip(client_losses, client_sizes)
        ) / sum(client_sizes)

        accuracy = evaluate_model(global_model, test_data)

        print(f"Round {round_num + 1} average loss: {avg_loss:.4f}")
        print(f"Round {round_num + 1} test accuracy: {accuracy:.2f}%")
        print(
            "Aggregation weights:",
            np.round(aggregation_weights, 3).tolist()
        )

        round_results.append((round_num + 1, avg_loss, accuracy))

    client_test_partitions = partition_test_like_train(
        train_data,
        test_data,
        clients,
        seed=seed
    )

    print("\n=== Per-Client Fairness Evaluation ===")

    client_accuracies = evaluate_clients(
        global_model,
        test_data,
        client_test_partitions
    )

    save_round_results("term", distribution, round_results)
    save_client_results("term", distribution, client_accuracies)

    print("\nTERM training complete!")
    return global_model


if __name__ == "__main__":
    model = train_term(
        rounds=2,
        num_clients=10,
        distribution="non_iid",
        alpha=0.5,
        tilt=1.0,
        seed=42
    )