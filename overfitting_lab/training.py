from copy import deepcopy
import torch
import torch.nn as nn

def evaluate_loss(model, x, y, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y)
    return float(loss.item())

def train_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=700,
    learning_rate=0.01,
    weight_decay=0.0,
    early_stopping=False,
    patience=40,
    min_delta=1e-4,
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    no_improve_count = 0

    for epoch in range(1, epochs + 1):
        model.train()

        logits = model(x_train)
        train_loss = criterion(logits, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        current_train_loss = float(train_loss.item())
        current_val_loss = evaluate_loss(
            model, x_val, y_val, criterion
        )

        train_losses.append(current_train_loss)
        val_losses.append(current_val_loss)

        if current_val_loss < best_val_loss - min_delta:
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if early_stopping and no_improve_count >= patience:
            break

    if early_stopping:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs_ran": len(train_losses),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_gap": val_losses[-1] - train_losses[-1],
    }
