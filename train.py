import torch
from model import SineNet, MSELoss, CustomSGD
import os

# ---------- Generate synthetic sine data ----------
torch.manual_seed(42)
x = torch.rand(500, 1) * 10
y = torch.sin(x) + torch.randn(500, 1) * 0.2

x_train, y_train = x[:400], y[:400]
x_test, y_test = x[400:], y[400:]

# ---------- Initialize model and optimizer ----------
model = SineNet()
optimizer = CustomSGD(model.parameters(), lr=0.01)

# ---------- Training loop ----------
for epoch in range(30000):
    preds = model(x_train)
    loss = MSELoss.apply(preds, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 3000 == 0:
        with torch.no_grad():
            test_preds = model(x_test)
            test_loss = MSELoss.apply(test_preds, y_test)
        print(f"Epoch: {epoch:5d} | Train Loss: {loss:.4f} | Test Loss: {test_loss:.4f}")

# ---------- Save the model ----------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/sine_model.pth")
print("Model saved to models/sine_model.pth")