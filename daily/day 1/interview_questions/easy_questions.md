# Day 1 - Easy Interview Questions

## 1. What is the relationship between Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL)?

**Answer:**
- **Artificial Intelligence (AI)** is the broadest concept - it refers to machines that can perform tasks that typically require human intelligence.
- **Machine Learning (ML)** is a subset of AI that enables machines to learn from data without being explicitly programmed.
- **Deep Learning (DL)** is a subset of Machine Learning that uses neural networks with multiple layers (hence "deep") to learn representations of data.

**Hierarchy:** AI ⊃ ML ⊃ DL

---

## 2. What are the three main types of machine learning?

**Answer:**
1. **Supervised Learning**: Learning from labeled data (input-output pairs). Examples: classification, regression.
2. **Unsupervised Learning**: Learning from unlabeled data to find patterns. Examples: clustering, dimensionality reduction.
3. **Reinforcement Learning**: Learning through interaction with an environment using rewards and penalties. Examples: game playing, robotics.

---

## 3. What is a neural network in simple terms?

**Answer:**
A neural network is a computational model inspired by the human brain. It consists of interconnected nodes (neurons) organized in layers. Each connection has a weight, and the network learns by adjusting these weights to minimize errors in predictions.

---

## 4. Why is Deep Learning called "deep"?

**Answer:**
Deep Learning is called "deep" because it uses neural networks with multiple hidden layers (typically more than 2-3 layers). The depth allows the network to learn hierarchical representations of data, where each layer learns increasingly complex features.

---

## 5. What is Google Colab and why is it useful for Deep Learning?

**Answer:**
Google Colab (Colaboratory) is a free cloud-based Jupyter notebook environment that provides:
- Free access to GPUs and TPUs
- Pre-installed libraries (PyTorch, TensorFlow, NumPy, etc.)
- No local installation required
- Easy sharing and collaboration
- Free tier with GPU access for deep learning projects

---

## 6. How do you check if PyTorch is installed and GPU is available?

**Answer:**
```python
import torch

# Check PyTorch version
print(f"PyTorch: {torch.__version__}")

# Check GPU availability
print(f"GPU Available: {torch.cuda.is_available()}")
```

---

## 7. What is a tensor?

**Answer:**
A tensor is a multi-dimensional array (generalization of matrices). In deep learning:
- 0D tensor = scalar (single number)
- 1D tensor = vector (1D array)
- 2D tensor = matrix (2D array)
- 3D+ tensor = higher-dimensional arrays

Tensors are the fundamental data structure used in PyTorch and TensorFlow.

---

## 8. What is the main advantage of using GPUs for Deep Learning?

**Answer:**
GPUs (Graphics Processing Units) have thousands of cores that can perform parallel computations simultaneously. This makes them much faster than CPUs for:
- Matrix multiplications (core operation in neural networks)
- Training large neural networks
- Processing large batches of data

GPUs can be 10-100x faster than CPUs for deep learning tasks.

---

## 9. What is the difference between PyTorch and TensorFlow?

**Answer:**
Both are deep learning frameworks, but with different approaches:
- **PyTorch**: 
  - Dynamic computation graphs (define-by-run)
  - More Pythonic and intuitive
  - Better for research and prototyping
  - Developed by Facebook
  
- **TensorFlow**:
  - Static computation graphs (define-then-run)
  - Better for production deployment
  - More comprehensive ecosystem
  - Developed by Google

---

## 10. What does it mean that neural networks are "universal function approximators"?

**Answer:**
The Universal Approximation Theorem states that a neural network with a single hidden layer containing a sufficient number of neurons can approximate any continuous function to arbitrary accuracy. This means neural networks can theoretically learn any complex relationship between inputs and outputs, given enough neurons and proper training.

