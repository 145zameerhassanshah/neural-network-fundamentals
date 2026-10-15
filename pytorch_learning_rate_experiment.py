import torch
import torch.nn as nn

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=torch.float32)
y = torch.tensor([[0.],[1.],[1.],[0.]], dtype=torch.float32)

class TinyXORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        return self.net(x)

def train_with_lr(lr, epochs=3000):
    torch.manual_seed(42)
    model = TinyXORNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    initial_loss = None

    for _ in range(epochs):
        logits = model(X)
        loss = criterion(logits, y)

        if initial_loss is None:
            initial_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return initial_loss, loss.item()

if __name__ == "__main__":
    print("=" * 72)
    print("PYTORCH LEARNING RATE EXPERIMENT")
    print("=" * 72)

    for lr in [0.1, 0.5, 1.0, 2.0]:
        initial_loss, final_loss = train_with_lr(lr)
        print(
            f"LR={lr:<3} | "
            f"Initial Loss={initial_loss:.6f} | "
            f"Final Loss={final_loss:.6f}"
        )
