import csv
import os


def save_round_results(algorithm, distribution, round_results):
    """
    Save training accuracy and loss for each round.
    """
    os.makedirs("results", exist_ok=True)

    filename = f"results/{algorithm}_{distribution}_rounds.csv"

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "round",
            "training_loss",
            "test_accuracy"
        ])

        for round_num, loss, accuracy in round_results:
            writer.writerow([round_num, loss, accuracy])

    print(f"Saved round results to {filename}")


def save_client_results(algorithm, distribution, accuracies):
    """
    Save accuracy for each client.
    """
    os.makedirs("results", exist_ok=True)

    filename = f"results/{algorithm}_{distribution}_clients.csv"

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["client_id", "accuracy"])

        for client_id, accuracy in enumerate(accuracies):
            writer.writerow([client_id, accuracy])

    print(f"Saved client results to {filename}")