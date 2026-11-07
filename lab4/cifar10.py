import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from loguru import logger

from config import PROJECT_ROOT, LAB4_ARTIFACT_DIR

DATA_DIR = PROJECT_ROOT / "data"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_plot(filename_stem: str):
    LAB4_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    art_path = LAB4_ARTIFACT_DIR / f"{filename_stem}.svg"
    plt.savefig(art_path, bbox_inches="tight", dpi=300)
    plt.close()


class CIFAR10MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 32 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.linear_relu_stack(x)


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()

    running_loss = 0.0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 200 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            logger.info(f"Batch {batch:3d}/{num_batches} -> loss: {loss_val:.4f} [{current:>5d}/{size:>5d}]")

    avg_loss = running_loss / num_batches
    accuracy = correct / size
    return avg_loss, accuracy


def evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return test_loss, correct


def main():
    torch.manual_seed(42)
    device = get_device()
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    logger.info(f"Używane urządzenie: {device} ({gpu_name})")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = CIFAR10MLP().to(device)
    logger.info(f"Architektura modelu:\n{model}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    epochs = 20
    history_rows = []

    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        logger.info(f"Start epoki {epoch}/{epochs}...")
        tr_loss, tr_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        te_loss, te_acc = evaluate(test_loader, model, loss_fn, device)
        scheduler.step()

        ep_time = time.perf_counter() - ep_start
        history_rows.append({
            "Epoka": epoch,
            "Train Loss": round(tr_loss, 4),
            "Test Loss": round(te_loss, 4),
            "Train Acc [%]": round(tr_acc * 100.0, 2),
            "Test Acc [%]": round(te_acc * 100.0, 2),
            "Czas [s]": round(ep_time, 2),
        })

        logger.info(
            f"Epoka {epoch:2d}/{epochs} ({ep_time:.1f}s) -> "
            f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc*100:.2f}% | "
            f"Test Loss: {te_loss:.4f}, Test Acc: {te_acc*100:.2f}%"
        )

    elapsed = time.perf_counter() - start_time
    df_history = pd.DataFrame(history_rows)
    best_acc = df_history["Test Acc [%]"].max()
    logger.info(f"Trening zakończony w {elapsed:.1f}s. Najlepsze Test Accuracy: {best_acc:.2f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    epochs_range = df_history["Epoka"]
    int_ticks = list(range(2, epochs + 1, 2))

    ax1.plot(epochs_range, df_history["Train Loss"], label="Zbiór treningowy", marker="o", markersize=3)
    ax1.plot(epochs_range, df_history["Test Loss"], label="Zbiór testowy", marker="s", markersize=3)
    ax1.set_xticks(int_ticks)
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("CIFAR-10 MLP Baseline: Funkcja straty")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs_range, df_history["Train Acc [%]"], label="Zbiór treningowy", marker="o", markersize=3)
    ax2.plot(epochs_range, df_history["Test Acc [%]"], label="Zbiór testowy", marker="s", markersize=3)
    ax2.set_xticks(int_ticks)
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Accuracy [%]")
    ax2.set_title("CIFAR-10 MLP Baseline: Dokładność klasyfikacji")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    save_plot("cifar10_mlp_baseline")
    logger.info("Zapisano wykres: cifar10_mlp_baseline.svg")

    LAB4_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = LAB4_ARTIFACT_DIR / "wyniki_cifar10.xlsx"
    df_history.to_excel(excel_path, index=False)
    logger.info(f"Zapisano tabelę wyników do: {excel_path}")

    weights_path = LAB4_ARTIFACT_DIR / "cifar10_mlp_baseline.pt"
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Zapisano wagi modelu do: {weights_path}")


if __name__ == "__main__":
    main()