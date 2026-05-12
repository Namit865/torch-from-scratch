# torch-from-scratch

**Building PyTorch's core components by hand — no `nn.Linear`, no `torch.optim`, just tensors and derivatives.**

This project implements the fundamental building blocks of deep learning **from scratch** using only `torch.Tensor` and `torch.autograd.Function`. I wrote custom forward/backward passes for a linear layer, SiLU activation, MSE loss, and a simple SGD optimizer. Then I used them to train a small neural network that learns to approximate a sine function — and deployed it as a live API.

It’s not about the sine wave; it’s about owning every gradient, every update, and knowing that I can modify any part of the computation graph when building custom architectures (like my future local LLM).

---

## 📁 Project Structure
torch-from-scratch/
├── README.md
├── model.py # All custom autograd functions + SineNet
├── train.py # Training script (uses model.py)
├── app.py # FastAPI inference server
├── requirements.txt # Dependencies
└── models/
└── sine_model.pth # Trained weights


- **`model.py`** – The engine. Contains:
  - `CustomLinear` (forward/backward for `y = xW^T + b`)
  - `SiLU` activation and its derivative
  - `MSELoss` with manual gradient
  - `CustomSGD` optimizer (zero_grad + step)
  - `SineNet` – a small neural net built **only** from these custom pieces
- **`train.py`** – Generates noisy sine data, trains `SineNet` using the custom components, and saves the model to `models/sine_model.pth`.
- **`app.py`** – A FastAPI server that loads the trained model and exposes a `/predict?x=<value>` endpoint.

---

## 🧠 Motivation

Most PyTorch tutorials hide the magic behind `nn.Linear` and `loss.backward()`. I wanted to understand **exactly** what happens when I call those lines. This project forced me to:

- Derive the gradients of matrix multiplication and element-wise activations.
- Implement reverse-mode autograd by hand (via `torch.autograd.Function`).
- Build an optimizer from scratch.
- Prove that I can design and train a neural net using no high-level NN wrappers.

It’s the foundation for building any custom architecture that doesn’t exist in a library — especially my own on-device LLM with non-standard attention or dynamic computation.

---

## 🚀 Quick Start

1. **Clone the repository**
   git clone https://github.com/Namit865/torch-from-scratch.git
   cd torch-from-scratch

2. **Install dependencies**
    pip install -r requirements.txt

3. ***Train the model** (optional – the repo already includes a trained model)
    python train.py

4. **Start the inference API**
    uvicorn app:app --reload

5. **Test the API**
   Visit http://127.0.0.1:8000/predict?x=0.2
    Example response:
    json
    {
      "Input": 0.2,
      "Output": 0.1673070788383484
    }

🔬 What Makes This Project Special

- ✅ Zero high-level wrappers – No nn.Linear, nn.MSELoss, torch.optim.
- ✅ Manual backward passes – Every gradient is explicitly coded using chain rule.
- ✅ Custom optimizer – Simple but functional SGD with zero_grad() and step().
- ✅ End-to-end training loop – Data generation, forward, loss, backward, update.
- ✅ Production-ready API – Model served via FastAPI for easy integration.

**📈 Training Output**
Epoch: 0      | Train Loss: 0.5632 | Test Loss: 0.5811
Epoch: 3000   | Train Loss: 0.0427 | Test Loss: 0.0445
Epoch: 6000   | Train Loss: 0.0391 | Test Loss: 0.0412
...
Epoch: 27000  | Train Loss: 0.0354 | Test Loss: 0.0361
Model Saved!

**🛠️ Built With**
PyTorch – Tensor operations + autograd.Function
FastAPI – Lightweight API wrapper
Uvicorn – ASGI server

**📚 Key Learnings**
- The backward() of a linear layer is simply grad_output @ weight (for input) and grad_output^T @ input (for weight).
- Sigmoid + its derivative appear consistently in activation functions like SiLU.
- An optimizer is just a loop over parameters: p.data -= lr * p.grad.
- PyTorch’s apply() method allows you to inject any custom function into the autograd graph, giving you full control.

🤝 Connect
If you're also digging into the internals of deep learning or building custom AI from scratch, I’d love to connect.
linkdin: www.linkedin.com/in/namit-senjaliya-68a3162a2
