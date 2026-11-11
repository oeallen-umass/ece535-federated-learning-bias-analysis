
from torchvision import datasets, transforms


def load_data():
    transform = transforms.ToTensor()

    train_data = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = load_data()

    print("Training images:", len(train_data))
    print("Testing images:", len(test_data))
    print("Image shape:", train_data[0][0].shape)
    print("First image label:", train_data[0][1])