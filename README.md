# torch-from-scratch: Building PyTorch's internals by hand

This project implements core deep learning primitives **without using high-level `torch.nn` or `torch.optim`** — I wrote my own `autograd.Function` subclasses for:

- Linear layer (forward + backward)
- SiLU activation (forward + backward)
- MSE loss (forward + backward)
- Custom SGD optimizer

Then I built a simple neural network (`SineNet`) using *only these custom building blocks* to approximate a sine wave, proving I understand exactly what happens under the hood of every PyTorch training loop.

**Why this matters:**  
I can now modify any part of the computation graph (loss, optimizer, layer) because I've implemented the gradients manually. This is the foundation for building custom AI architectures that don't exist in any library.

**File Structure:**
├── README.md
├── model.py          # Contains all custom classes + SineNet
├── train.py          # Training script
├── app.py            # FastAPI inference
├── requirements.txt  # torch, fastapi, uvicorn
└── models/
    └── sine_model.pth

## Quick preview
- Trained a 16→16→1 network from scratch (no `nn.Linear`)
- Deployed with FastAPI for live predictions
- Check out the [training curves](#) ...
