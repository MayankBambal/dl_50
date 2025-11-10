# Day 1 - Deep Learning Fundamentals Cheat Sheet

## 🎯 AI/ML/DL Hierarchy

```
Artificial Intelligence (AI)
    └── Machine Learning (ML)
            └── Deep Learning (DL)
```

- **AI**: Machines performing tasks requiring human intelligence
- **ML**: Learning from data without explicit programming
- **DL**: ML using multi-layer neural networks

---

## 📚 Types of Machine Learning

| Type | Description | Example |
|------|-------------|---------|
| **Supervised** | Learn from labeled data (input-output pairs) | Email spam detection, image classification |
| **Unsupervised** | Find patterns in unlabeled data | Customer segmentation, anomaly detection |
| **Reinforcement** | Learn through rewards/penalties in an environment | Game AI, robotics, autonomous vehicles |

---

## 🧠 Neural Networks Basics

### Key Concepts
- **Neuron/Node**: Basic processing unit
- **Layer**: Collection of neurons
  - Input layer: Receives data
  - Hidden layer(s): Process data
  - Output layer: Produces predictions
- **Weight**: Connection strength between neurons
- **Bias**: Additional parameter for flexibility
- **Activation Function**: Introduces non-linearity

### Universal Function Approximator
- Neural networks can approximate any continuous function
- Requires sufficient neurons and proper training
- Enables learning complex input-output relationships

---

## 🔧 Deep Learning Characteristics

### What Makes It "Deep"?
- Multiple hidden layers (typically >2-3)
- Hierarchical feature learning:
  - **Early layers**: Low-level features (edges, textures)
  - **Middle layers**: Mid-level features (shapes, patterns)
  - **Deep layers**: High-level features (objects, concepts)

### Key Advantages
- ✅ Automatic feature learning (no manual engineering)
- ✅ Handles complex, non-linear patterns
- ✅ Scales well with data
- ✅ State-of-the-art performance on many tasks

### Key Disadvantages
- ❌ Requires large amounts of data
- ❌ Computationally expensive (needs GPUs)
- ❌ Less interpretable ("black box")
- ❌ Longer training times

---

## 🛠️ Tools & Setup

### Google Colab
```python
# Access: https://colab.research.google.com
# Enable GPU: Runtime → Change runtime type → GPU
```

**Advantages:**
- Free GPU/TPU access
- Pre-installed libraries
- No local installation
- Easy sharing

**Limitations:**
- Session timeouts
- Temporary storage
- Resource limits

### PyTorch Setup & Verification
```python
import torch

# Check version
print(torch.__version__)

# Check GPU availability
print(torch.cuda.is_available())

# Get GPU name (if available)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### TensorFlow Setup & Verification
```python
import tensorflow as tf

# Check version
print(tf.__version__)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)}")

# List GPU devices
for gpu in gpus:
    print(gpu)
```

### Basic GPU Test
```python
# PyTorch GPU test
x = torch.randn(100, 100).cuda()
y = torch.randn(100, 100).cuda()
z = torch.matmul(x, y)
print(f"Device: {z.device}")

# TensorFlow GPU test
with tf.device('/GPU:0'):
    a = tf.random.normal((100, 100))
    b = tf.random.normal((100, 100))
    c = tf.matmul(a, b)
print(f"Device: {c.device}")
```

---

## 📊 Tensors

### Definition
Multi-dimensional arrays (generalization of matrices)

### Dimensions
- **0D Tensor**: Scalar (single number)
- **1D Tensor**: Vector (1D array)
- **2D Tensor**: Matrix (2D array)
- **3D+ Tensor**: Higher-dimensional arrays

### Example
```python
# 0D: scalar
scalar = torch.tensor(5)

# 1D: vector
vector = torch.tensor([1, 2, 3])

# 2D: matrix
matrix = torch.tensor([[1, 2], [3, 4]])

# 3D: cube
cube = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

---

## ⚡ GPU Acceleration

### Why GPUs?
- **Parallel Processing**: Thousands of cores vs. CPU's few cores
- **Matrix Operations**: Highly parallelizable (core of neural networks)
- **Speed**: 10-100x faster for deep learning tasks

### Key Operations
- Matrix multiplication: `Y = X × W + b`
- Convolution operations
- Batch processing

### When to Use GPU
- ✅ Training large models
- ✅ Processing large batches
- ✅ Complex architectures
- ❌ Small models (overhead may not be worth it)
- ❌ Inference on single samples (CPU often sufficient)

---

## 🔄 PyTorch vs TensorFlow

| Feature | PyTorch | TensorFlow |
|---------|---------|------------|
| **Graph Type** | Dynamic (define-by-run) | Static (define-then-run) |
| **Ease of Use** | More Pythonic, intuitive | Steeper learning curve |
| **Best For** | Research, prototyping | Production deployment |
| **Debugging** | Easier (eager execution) | More complex |
| **Deployment** | Good | Excellent (TF Serving, Lite) |
| **Community** | Research-focused | Industry-focused |
| **Developer** | Facebook (Meta) | Google |

### When to Choose PyTorch
- Research and experimentation
- Dynamic architectures
- Python-first development
- Academic projects

### When to Choose TensorFlow
- Production deployment
- Mobile/edge devices
- Large-scale distributed training
- Google Cloud integration

---

## 🎓 Key Concepts

### Training vs Inference
- **Training**: Learning parameters from data (forward + backward pass)
  - Computationally intensive
  - Requires GPUs
  - Done once/periodically
  
- **Inference**: Making predictions with trained model (forward pass only)
  - Much faster
  - Can run on CPU
  - Done repeatedly in production

### End-to-End Learning
- Model learns directly from raw input to output
- No manual feature engineering
- Single network handles entire pipeline
- Requires more data but learns optimal features

### Computation Graph
- Represents mathematical operations as nodes
- Edges represent data flow
- Enables automatic differentiation (backpropagation)
- **Static**: Defined before execution (TensorFlow)
- **Dynamic**: Built during execution (PyTorch)

---

## 📝 Common Libraries

### Essential Libraries
```python
import torch          # Deep learning framework
import tensorflow as tf # Alternative framework
import numpy as np     # Numerical computing
import pandas as pd    # Data manipulation
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Statistical visualization
```

### Version Checking
```python
import torch
import tensorflow as tf
import numpy as np
import pandas as pd

print(f"PyTorch: {torch.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
```

---

## 🚀 Quick Reference Commands

### Colab Setup
1. Go to: https://colab.research.google.com
2. New notebook: File → New notebook
3. Enable GPU: Runtime → Change runtime type → Hardware accelerator: GPU
4. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`

### Essential Checks
```python
# PyTorch
import torch
torch.cuda.is_available()  # True if GPU available
torch.cuda.get_device_name(0)  # GPU name

# TensorFlow
import tensorflow as tf
tf.config.list_physical_devices('GPU')  # List GPUs
tf.test.is_gpu_available()  # Check GPU availability
```

---

## 💡 Key Takeaways

1. **AI ⊃ ML ⊃ DL**: Deep Learning is a subset of Machine Learning, which is a subset of AI

2. **Three Learning Types**: Supervised (labeled), Unsupervised (patterns), Reinforcement (rewards)

3. **Neural Networks**: Interconnected neurons in layers that learn hierarchical features

4. **Deep = Multiple Layers**: Enables learning complex, hierarchical representations

5. **GPUs Essential**: Parallel processing makes training feasible for large models

6. **Frameworks**: PyTorch (research) vs TensorFlow (production) - both powerful

7. **Tensors**: Fundamental data structure (multi-dimensional arrays)

8. **End-to-End**: Learn directly from raw data without manual feature engineering

9. **Colab**: Great starting point with free GPU access

10. **Universal Approximators**: Neural networks can learn any continuous function

---

## 📖 Further Reading

- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 1
- **Course**: Andrew Ng's Deep Learning Specialization - Course 1, Week 1
- **Tutorial**: [PyTorch Getting Started](https://pytorch.org/get-started/locally/)
- **Tutorial**: [Google Colab Introduction](https://colab.research.google.com/notebooks/intro.ipynb)
- **Resource**: [fast.ai Deep Learning Course](https://www.fast.ai/)

