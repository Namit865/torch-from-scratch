from fastapi import FastAPI
import torch
from model import SineNet
import os

app = FastAPI()

# Load trained model
model = SineNet()
model_path = "models/sine_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

@app.get("/predict")
def predict(x: float):
    with torch.no_grad():
        inp = torch.tensor([[x]])
        out = model(inp)
    return {"Input": x, "Output": out.item()}