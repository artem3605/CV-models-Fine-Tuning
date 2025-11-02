"""
Training script for CLIP Embedding-based classification.
"""

import os
import sys
from contextlib import nullcontext, redirect_stdout

import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prepare_dataset import get_dataset
from clip_experiments.clip_embeddings import (
    CLIPEmbeddingClassifier,
    CLIPDatasetWrapper,
    apply_clip_embedding_strategy
)

# Constants
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
TRAINING_LOGS_DIR = "./training_logs_clip/"
CONFIG_PATH = "./clip_experiments/config_clip.yaml"


def read_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def training_pipeline_clip(
    model: torch.nn.Module,
    dataset,
    device: str,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    num_epochs: int,
    mixed_precision: bool,
    save_path: str,
    save: bool = False,
    training_strategy: str = "zero_shot"
):
    """Training pipeline for CLIP embedding models."""

    model_name = f"CLIPEmbedding_{training_strategy}"

    log_path = os.path.join(TRAINING_LOGS_DIR, f"{model_name}.log")
    ensure_dir(TRAINING_LOGS_DIR)

    with open(log_path, "w", buffering=1) as log_file, redirect_stdout(log_file):
        print(f"=== Training CLIP Embeddings with strategy: {training_strategy} ===")
        print(f"Device: {device}, LR: {lr}, Batch size: {batch_size}, Epochs: {num_epochs}")

        clip_dataset = CLIPDatasetWrapper(dataset, model.processor)

        train_size = int(TRAIN_SPLIT * len(clip_dataset))
        val_size = int(VALIDATION_SPLIT * len(clip_dataset))
        test_size = int(TEST_SPLIT * len(clip_dataset))

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            clip_dataset, [train_size, val_size, test_size]
        )

        # Data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        model = model.to(device)
        model = apply_clip_embedding_strategy(model, training_strategy)

        if training_strategy == "zero_shot":
            print("\n[Zero-Shot] Skipping training, evaluating directly...\n")
            num_epochs = 0

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        if len(trainable_params) > 0:
            optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = None

        criterion = torch.nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda") if (mixed_precision and device == "cuda") else None

        best_loss = float("inf")
        wait = 0

        train_loss_hist, train_acc_hist = [], []
        val_loss_hist, val_acc_hist = [], []

        # Training loop
        for epoch in range(1, num_epochs + 1):
            if optimizer is not None:
                model.train()
                running_loss, running_correct, total = 0.0, 0, 0

                autocast_ctx = (
                    torch.amp.autocast(device_type="cuda")
                    if (mixed_precision and device == "cuda")
                    else nullcontext()
                )

                # Training
                for images, labels in tqdm(
                    train_loader, desc=f"[{model_name}] Epoch {epoch}", leave=False
                ):
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

                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    running_correct += (predicted == labels).sum().item()

                epoch_loss = running_loss / total
                epoch_acc = running_correct / total
                train_loss_hist.append(epoch_loss)
                train_acc_hist.append(epoch_acc)

                print(f"[{model_name}] Train Epoch {epoch}: Loss={epoch_loss:.5f}, Acc={epoch_acc:.5f}")

                # Update text embeddings after training text encoder
                if training_strategy in ["both_encoders", "text_and_image", "lora"]:
                    model.update_text_embeddings()

            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)

                    val_loss += criterion(outputs, labels).item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total
            val_loss_hist.append(val_epoch_loss)
            val_acc_hist.append(val_epoch_acc)

            if num_epochs > 0:
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

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_epoch_acc = test_correct / test_total
        print(f"\n[{model_name}] Final Test Accuracy: {test_epoch_acc:.5f}\n")

        if save and num_epochs > 0:
            checkpoint_dir = os.path.dirname(save_path)
            if checkpoint_dir:
                ensure_dir(checkpoint_dir)
            torch.save(model.state_dict(), save_path)
            print(f"[{model_name}] Model saved to {save_path}")

        print(f"=== Training finished for {model_name} ===\n")

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, test_epoch_acc


def plot_results(train_loss, train_acc, val_loss, val_acc, model_name: str):
    """Plot training curves."""
    if len(train_loss) == 0:
        print(f"No training data to plot for {model_name} (zero-shot)")
        return

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
    plt.savefig(f"./training_logs_clip/{model_name}_curves.png")
    plt.close()


if __name__ == "__main__":
    config = read_config()
    ensure_dir(TRAINING_LOGS_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    dataset = get_dataset(config["image_size"])
    print(f"Dataset size: {len(dataset)}")

    class_names = config["class_names"]
    print(f"Class names: {class_names}\n")

    strategies = config["clip_strategies"]

    results = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Training strategy: {strategy}")
        print(f"{'='*60}\n")

        # Create fresh model for each strategy
        model = CLIPEmbeddingClassifier(
            class_names=class_names,
            templates=config.get("templates", None)
        )

        model_key = f"CLIPEmbedding_{strategy}"

        train_loss, train_acc, val_loss, val_acc, test_acc = training_pipeline_clip(
            model,
            dataset,
            device,
            config["learning_rate"],
            config["weight_decay"],
            config["batch_size"],
            config["patience"],
            config["num_epochs"],
            config["mixed_precision"],
            os.path.join(config["checkpoint_path"], f"{model_key}.pt"),
            config["save_checkpoint"],
            strategy
        )

        results[model_key] = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_acc": test_acc,
        }

        # Plot results
        if len(train_loss) > 0:
            plot_results(train_loss, train_acc, val_loss, val_acc, model_key)

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for strategy, data in results.items():
        print(f"{strategy}: Test Accuracy = {data['test_acc']:.5f}")
    print("="*60)

    print(f"\nTraining complete! Logs saved to {TRAINING_LOGS_DIR}")