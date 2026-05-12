```markdown
# torch-from-scratch

**Building PyTorch's internals by hand — no `nn.Linear`, no `nn.MSELoss`, no `torch.optim`.**

This project implements core deep learning primitives from scratch using raw `torch.autograd.Function`:
- Custom Linear layer (forward + backward)
- SiLU activation (forward + backward)
- MSE loss (forward + backward)
- Custom SGD optimizer

A simple neural network (`SineNet`) is built using *only* these custom components and trained to approximate a sine wave. Finally, a FastAPI endpoint serves live predictions.

> The goal is not the sine wave — it's proving deep understanding of what really happens inside a training loop. This is the foundation for designing custom AI architectures that no library can provide out of the box.

---

## 📁 Project Structure

```
torch-from-scratch/
├── model.py          # All custom building blocks + SineNet
├── train.py          # Generates data, trains the model, saves weights
├── app.py            # FastAPI server for live predictions
├── requirements.txt  # Only the necessary packages
├── .gitignore
├── README.md
└── models/           # Created automatically; contains trained weights
    └── sine_model.pth
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/torch-from-scratch.git
cd torch-from-scratch
```

### 2. Set up a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate     # macOS / Linux
# or
venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train.py
```
This will print the training progress and save the model to `models/sine_model.pth`.

### 5. Start the API server
```bash
uvicorn app:app --reload
```

### 6. Test the endpoint
Open your browser and visit:
```
http://127.0.0.1:8000/predict?x=0.2
```
You’ll get a JSON response like:
```json
{"Input":0.2,"Output":0.1673}
```

Try other values by changing `?x=`.

---

## 🧠 Why this project exists

Most PyTorch tutorials teach you to use `nn.Linear`, `F.mse_loss`, and `torch.optim.Adam` as black boxes. That's fine for building quickly, but not for **inventing something new**.

Here I explicitly wrote the forward and backward passes, the loss gradient, and the parameter update rules. Now I can:
- Modify any part of the training loop at the mathematical level
- Design novel layers with custom gradients
- Build a fully custom LLM or any architecture without being constrained by existing library implementations

If you're on the same path of "understanding before using", this repo is for you.

---

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- FastAPI
- Uvicorn

All listed in `requirements.txt`.

---

## 📬 Try it live

After training and starting the server, explore different inputs:
- `http://127.0.0.1:8000/predict?x=3.14`
- `http://127.0.0.1:8000/predict?x=6.28`
- Even negative numbers: `?x=-1.5`

The model learned to approximate `sin(x)`, so it will output values roughly between -1 and 1.

---

## ✍️ Author

Built by Namit Senjaliya while digging deep into PyTorch internals.  
Connect with me on LinkedIn www.linkedin.com/in/namit-senjaliya-68a3162a2
