
import torch
import torch.nn as nn


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = FashionCNN()

    # Test the model with 4 fake images
    sample = torch.randn(4, 1, 28, 28)
    output = model(sample)

    print(model)
    print("\nInput shape:", sample.shape)
    print("Output shape:", output.shape)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters())
    )