from contextlib import nullcontext, redirect_stdout
import torch
import yaml
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import VisionDataset

from prepare_dataset import get_dataset
from models import models, apply_training_strategy
import copy

# Constants
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
TRAINING_LOGS_DIR = "./training_logs/"
CONFIG_PATH = "./config.yaml"

# Utils
def read_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def training_pipeline(
    model: torch.nn.Module,
    dataset: VisionDataset,
    device: str,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    num_epochs: int,
    mixed_precision: bool,
    save_path: str,
    save: bool = False,
    training_strategy: str = "full"
):
    model_name = model.__class__.__name__

    log_path = os.path.join(TRAINING_LOGS_DIR, f"{model_name}_{training_strategy}.log")
    ensure_dir(TRAINING_LOGS_DIR)

    with open(log_path, "w", buffering=1) as log_file, redirect_stdout(log_file):
        print(f"=== Training started for {model_name} with strategy: {training_strategy} ===")
        print(f"Device: {device}, LR: {lr}, Batch size: {batch_size}, Epochs: {num_epochs}")

        # Dataset split
        train_size = int(TRAIN_SPLIT * len(dataset))
        val_size = int(VALIDATION_SPLIT * len(dataset))
        test_size = int(TEST_SPLIT * len(dataset))

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        model = model.to(device)
        model = apply_training_strategy(model, training_strategy)

        # Only optimize parameters that require gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

        criterion = torch.nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda") if (mixed_precision and device == "cuda") else None

        best_loss = float("inf")
        wait = 0

        train_loss_hist, train_acc_hist = [], []
        val_loss_hist, val_acc_hist = [], []

        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss, running_correct, total = 0.0, 0, 0

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda") if (mixed_precision and device == "cuda") else nullcontext()
            )

            # Training
            for images, labels in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}", leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast_ctx:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # Gradient health check
                grad_mean = sum(
                    p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None
                ) / sum(1 for p in model.parameters() if p.grad is not None)
                if grad_mean < 1e-4:
                    print(f"[{model_name}] Warning: very low grad mean ({grad_mean:.2e})")

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                running_correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = running_correct / total
            train_loss_hist.append(epoch_loss)
            train_acc_hist.append(epoch_acc)

            print(f"[{model_name}] Train Epoch {epoch}: Loss={epoch_loss:.5f}, Acc={epoch_acc:.5f}")

            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    with autocast_ctx:
                        outputs = model(images)

                    val_loss += criterion(outputs, labels).item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total
            val_loss_hist.append(val_epoch_loss)
            val_acc_hist.append(val_epoch_acc)

            print(f"[{model_name}] Val Epoch {epoch}: Loss={val_epoch_loss:.5f}, Acc={val_epoch_acc:.5f}")

            # Early stopping
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"[{model_name}] Early stopping on epoch {epoch}")
                    break

        # Testing
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                with autocast_ctx:
                    outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_epoch_acc = test_correct / test_total
        print(f"Accuracy on Test: {test_epoch_acc:.5f}")

        if save:
            torch.save(model.state_dict(), save_path)
            print(f"[{model_name}] Model saved to {save_path}")

        print(f"=== Training finished for {model_name} ===\n")

        # Vanishing
        del model
        torch.cuda.empty_cache()

        return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, epoch


def plot_results(train_loss: list[float], train_acc: list[float],
                 val_loss: list[float], val_acc: list[float],
                 model_name: str):
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Training curves for {model_name}", fontsize=14)

    ax[0].plot(epochs, train_loss, label="Train Loss")
    ax[0].plot(epochs, val_loss, label="Validation Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(epochs, train_acc, label="Train Accuracy")
    ax[1].plot(epochs, val_acc, label="Validation Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = read_config()
    ensure_dir(TRAINING_LOGS_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = get_dataset(config["image_size"])
    print(f"Dataset size: {len(dataset)}")

    results = {}

    training_strategy = config["training_strategy"]

    for model in models:
        model_copy = copy.deepcopy(model)

        print(f"\n=== Training {model_copy.__class__.__name__} ===\n")
        model_key = f"{model_copy.__class__.__name__}_{training_strategy}"

        results[model_key] = training_pipeline(
            model_copy,
            dataset,
            device,
            config["learning_rate"],
            config["weight_decay"],
            config["batch_size"],
            config["patience"],
            config["num_epochs"],
            config["mixed_precision"],
            config["checkpoint_path"],
            config["save_checkpoint"],
            training_strategy
        )

    # Plot results
    for name, (train_loss, train_acc, val_loss, val_acc, epoch) in results.items():
        plot_results(train_loss, train_acc, val_loss, val_acc, name)

    print("That's it! You can find logs in ", TRAINING_LOGS_DIR)
