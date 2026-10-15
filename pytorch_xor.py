import torch
import torch.nn as nn

torch.manual_seed(42)

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=torch.float32)
y = torch.tensor([[0.],[1.],[1.],[0.]], dtype=torch.float32)

class TinyXORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        return self.output(x)  # raw logits

model = TinyXORNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

epochs = 5000

print("=" * 70)
print("PYTORCH XOR TRAINING")
print("=" * 70)

for epoch in range(1, epochs + 1):
    logits = model(X)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 500 == 0:
        print(f"Epoch {epoch:04d} | BCE Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    logits = model(X)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()

print("\nFinal predictions:")
for features, target, probability, predicted in zip(X, y, probabilities, predictions):
    print(
        f"Input {features.tolist()} | "
        f"Target {int(target.item())} | "
        f"Probability {probability.item():.4f} | "
        f"Class {int(predicted.item())}"
    )
