import numpy as np
import pandas as pd
import torch


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y, lengths in data_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def validate_one_epoch(
    model,
    data_loader,
    criterion,
    device,
    calculate_metrics,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_probabilities = []

    for x, y, lengths in data_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x, lengths)
        loss = criterion(logits, y)
        probabilities = torch.sigmoid(logits)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(y.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probabilities)

    return (
        total_loss / total_samples,
        calculate_metrics(y_true, y_prob),
    )


def fit(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    callbacks,
    device,
    calculate_metrics,
    epochs=30,
):
    history = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        valid_loss, valid_metrics = validate_one_epoch(
            model,
            valid_loader,
            criterion,
            device,
            calculate_metrics,
        )

        val_auc = valid_metrics["auc"]
        val_f1_macro = valid_metrics["f1_macro"]

        callbacks["scheduler"].step(val_auc)
        callbacks["checkpoint"].update(
            model,
            val_auc,
            val_f1_macro,
        )
        callbacks["early_stopping"](val_auc, model)

        writer = callbacks["writer"]
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/valid", valid_loss, epoch + 1)
        writer.add_scalar("AUC/valid", val_auc, epoch + 1)
        writer.add_scalar("F1_macro/valid", val_f1_macro, epoch + 1)
        writer.add_scalar("F1_micro/valid", valid_metrics["f1_micro"], epoch + 1)
        writer.add_scalar(
            "Learning_rate",
            optimizer.param_groups[0]["lr"],
            epoch + 1,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            **valid_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"AUC: {val_auc:.4f} | "
            f"F1 Macro: {val_f1_macro:.4f} | "
            f"F1 Micro: {valid_metrics['f1_micro']:.4f}"
        )

        if callbacks["early_stopping"].early_stop:
            break

    callbacks["early_stopping"].restore_best_weights(model)
    callbacks["writer"].close()

    return model, pd.DataFrame(history)

