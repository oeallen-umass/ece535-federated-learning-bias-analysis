
import numpy as np
from torchvision import datasets
from data import load_data


def partition_iid(dataset, num_clients=10, seed=42):
    """Randomly divide data equally among clients."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))

    return [
        split.tolist()
        for split in np.array_split(indices, num_clients)
    ]


def partition_non_iid(dataset, num_clients=10, alpha=0.5, seed=42):
    """
    Divide data using a Dirichlet distribution.
    Lower alpha means clients have more different
    class distributions.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset.targets)
    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(
            np.full(num_clients, alpha)
        )

        split_points = (
            np.cumsum(proportions)[:-1] * len(class_indices)
        ).astype(int)

        splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    for indices in client_indices:
        rng.shuffle(indices)

    return client_indices


def print_client_distribution(dataset, partitions):
    """Display the number of samples per class for each client."""
    labels = np.asarray(dataset.targets)

    print("\nClient data distributions:")
    print("Client | Total | Class counts (0-9)")
    print("-" * 70)

    for client_id, indices in enumerate(partitions):
        counts = np.bincount(
            labels[indices],
            minlength=10
        )

        print(
            f"{client_id:6d} | "
            f"{len(indices):5d} | "
            f"{counts.tolist()}"
        )


if __name__ == "__main__":
    train_data, _ = load_data()

    print("=== IID Partition ===")
    iid_clients = partition_iid(train_data)
    print_client_distribution(train_data, iid_clients)

    print("\n=== Non-IID Partition ===")
    non_iid_clients = partition_non_iid(train_data)
    print_client_distribution(train_data, non_iid_clients)