# Installation Guide

Complete setup instructions for the 50 Days of Deep Learning course.

## System Requirements

- **Operating System:** macOS, Linux, or Windows
- **Python Version:** Python 3.12 or higher (latest stable version recommended)
- **PyTorch Version:** PyTorch 2.5.0+ (latest stable)
- **RAM:** 8GB minimum (16GB recommended for later weeks)
- **GPU:** Optional but recommended for faster training (CUDA-compatible GPU for PyTorch)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/50-days-deep-learning.git
cd 50-days-deep-learning
```

### 2. Create a Virtual Environment

**Using venv (Recommended):**

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**Using conda (Alternative):**

```bash
conda create -n dl50 python=3.12
conda activate dl50
```

### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### 4. Install Jupyter (if needed)

```bash
pip install jupyter jupyterlab
```

### 5. Verify Installation

Test your installation:

```python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"Python version: {__import__('sys').version}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"NumPy version: {np.__version__}")
print("✅ All libraries installed successfully!")
```

## GPU Setup (Optional)

### For PyTorch (CUDA/MPS)

PyTorch supports both NVIDIA CUDA and Apple Metal Performance Shaders (MPS) for GPU acceleration.

**Check GPU availability:**

```python
import torch

# Check CUDA (NVIDIA GPUs)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Check MPS (Apple Silicon - M1/M2/M3)
print(f"MPS available: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    print(f"MPS built: {torch.backends.mps.is_built()}")
```

**Install CUDA-enabled PyTorch (if needed):**

Visit [PyTorch Installation Page](https://pytorch.org/get-started/locally/) to get the latest installation command for your system:

```bash
# Example for CUDA 12.4 (check latest version on PyTorch website)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Note:** PyTorch on Apple Silicon (M1/M2/M3) automatically uses MPS, no additional setup needed.

## Troubleshooting

### Issue: pip install fails

**Solution:**
```bash
pip install --upgrade pip setuptools wheel
```

### Issue: Import errors

**Solution:**
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Issue: Jupyter not found

**Solution:**
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=dl50
```

### Issue: Out of memory errors

**Solution:**
- Reduce batch size in exercises
- Use CPU instead of GPU
- Close other applications

## IDE Setup

### VS Code

1. Install Python extension
2. Select Python interpreter: `Cmd+Shift+P` → "Python: Select Interpreter"
3. Install Jupyter extension for notebook support

### PyCharm

1. Open project in PyCharm
2. Go to Settings → Project → Python Interpreter
3. Select your virtual environment

## Docker Setup (Advanced)

If you prefer using Docker:

```dockerfile
# Dockerfile example
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
```

## Next Steps

Once installation is complete:

1. Navigate to [Day 1](../daily/day_01/)
2. Read the README for that day
3. Open the Jupyter notebook
4. Start learning! 🚀

## Need Help?

- Check [FAQ](faq.md)
- Open an issue on GitHub
- Check the course discussions

