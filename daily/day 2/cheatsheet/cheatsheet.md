# Day 2 - The Perceptron Cheat Sheet

## 🎯 What is a Perceptron?

The **perceptron** is the simplest form of an artificial neuron, introduced by Frank Rosenblatt in 1957. It's a linear classifier that makes binary decisions (yes/no, 0/1, true/false).

### Key Characteristics
- ✅ Simplest neural network
- ✅ Linear classifier
- ✅ Binary output (0 or 1)
- ✅ Foundation for understanding complex neural networks

---

## 🏛️ Architecture

### Components
```
Inputs (x₁, x₂, ..., xₙ)
    ↓
Weights (w₁, w₂, ..., wₙ)
    ↓
Weighted Sum: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
    ↓
Activation Function (Step Function)
    ↓
Output (0 or 1)
```

### Mathematical Formula
**Weighted Sum:**
$$z = (w_1x_1 + w_2x_2 + ... + w_nx_n) + b$$

**Vector Notation:**
$$z = w \cdot x + b$$

**Activation (Step Function):**
$$\text{output} = \begin{cases} 
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}$$

---

## 🧠 Key Concepts

### 1. Weights (w)
- Represent the **importance** of each input
- Larger weight = more influence on decision
- Learned during training

### 2. Bias (b)
- Acts as a **threshold**
- Shifts the decision boundary left or right
- Allows flexibility in classification

### 3. Weighted Sum (z)
- Linear combination of inputs and weights
- `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
- Also called "logit" or "pre-activation"

### 4. Activation Function
- **Step Function (Heaviside)**: Converts weighted sum to binary output
- Forces perceptron to make definitive choice (0 or 1)

---

## 📈 Training: Perceptron Learning Rule

### Algorithm Steps
1. **Initialize**: Start with small random weights (or zeros) and bias
2. **For each training example:**
   - Calculate weighted sum: `z = w·x + b`
   - Make prediction: `y_pred = step_function(z)`
   - Calculate error: `error = y_true - y_pred`
   - Update weights: `w_new = w_old + (learning_rate × error × x)`
   - Update bias: `b_new = b_old + (learning_rate × error)`

### Update Rules
```python
# Weight update
w_new = w_old + (learning_rate * error * input)

# Bias update
b_new = b_old + (learning_rate * error)
```

### Intuition
- **Correct prediction** (error = 0): No change
- **Predict 0, should be 1** (error = +1): Add fraction of input to weights
- **Predict 1, should be 0** (error = -1): Subtract fraction of input from weights

### Learning Rate
- Controls step size of weight updates
- Small value (e.g., 0.01): Slow but stable learning
- Large value (e.g., 0.1): Fast but may overshoot

---

## 💻 Implementation

### Basic Perceptron Class
```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0
    
    def _step_function(self, x):
        """Heaviside step function"""
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Forward pass
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._step_function(linear_output)
                
                # Calculate error
                error = y[idx] - y_predicted
                
                # Update weights and bias
                update = self.learning_rate * error
                self.weights += update * x_i
                self.bias += update
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._step_function(linear_output)
```

---

## 📊 Decision Boundary

### What is it?
A **decision boundary** is the line (or hyperplane) that separates different classes.

### For 2D Input
- Equation: `w₁x₁ + w₂x₂ + b = 0`
- Can be rewritten as: `x₂ = (-w₁x₁ - b) / w₂`
- Visualizes how perceptron classifies data

### Visualization
```python
import matplotlib.pyplot as plt

# Plot data points
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], label='Class 0')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], label='Class 1')

# Plot decision boundary
x1_min_max = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
x2_boundary = (-weights[0] * x1_min_max - bias) / weights[1]
plt.plot(x1_min_max, x2_boundary, 'k--', label='Decision Boundary')
```

---

## ❌ Key Limitation: Linear Separability

### The Problem
**Perceptron can only solve linearly separable problems.**

A problem is **linearly separable** if a single straight line (or hyperplane) can perfectly separate the two classes.

### Examples

#### ✅ Linearly Separable: OR Gate
| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

**Visualization**: Can draw a line separating (0,0) from the other three points.

#### ❌ NOT Linearly Separable: XOR Gate
| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Visualization**: Cannot draw a single line to separate 0s and 1s.

### Perceptron Convergence Theorem
- **If linearly separable**: Algorithm **guaranteed** to find separating line
- **If NOT linearly separable**: Algorithm will **never converge** (loops forever)

---

## 🔑 Key Takeaways

1. **Simplest Neural Network**: Perceptron is the foundation for understanding complex networks

2. **Linear Classifier**: Can only learn linear decision boundaries

3. **Binary Output**: Uses step function to produce 0 or 1

4. **Learning Rule**: Updates weights based on prediction error

5. **Linear Separability**: Major limitation - cannot solve XOR problem

6. **Foundation for MLPs**: This limitation led to Multi-Layer Perceptrons (MLPs) with hidden layers

---

## 📝 Common Operations

### Forward Pass
```python
# Calculate weighted sum
z = np.dot(X, weights) + bias

# Apply activation
output = step_function(z)
```

### Training Loop
```python
for epoch in range(n_iters):
    for i, x_i in enumerate(X):
        # Predict
        y_pred = step_function(np.dot(x_i, weights) + bias)
        
        # Calculate error
        error = y[i] - y_pred
        
        # Update
        weights += learning_rate * error * x_i
        bias += learning_rate * error
```

### Prediction
```python
# Single prediction
prediction = perceptron.predict(X_new)

# Batch prediction
predictions = perceptron.predict(X_batch)
```

---

## 🎓 Historical Context

- **1957**: Frank Rosenblatt introduces perceptron
- **1969**: Minsky & Papert prove XOR limitation
- **1980s**: Multi-layer perceptrons (MLPs) overcome limitation
- **Today**: Foundation for understanding modern deep learning

---

## 🔗 Connection to Modern Deep Learning

The perceptron introduces core concepts used in all neural networks:
- ✅ Weights and biases
- ✅ Weighted sums
- ✅ Activation functions
- ✅ Learning through error correction
- ✅ Forward propagation

**Next Step**: Multi-Layer Perceptrons (MLPs) add hidden layers and non-linear activations to solve non-linearly separable problems like XOR.

---

## 📖 Quick Reference

### Formula Summary
- **Weighted Sum**: `z = w·x + b`
- **Activation**: `output = 1 if z ≥ 0, else 0`
- **Weight Update**: `w = w + η × error × x`
- **Bias Update**: `b = b + η × error`

### Key Parameters
- **Learning Rate (η)**: Step size for updates (typically 0.01-0.1)
- **Epochs (n_iters)**: Number of passes over training data
- **Weights**: Learned parameters (one per input feature)
- **Bias**: Single learned threshold parameter

---

## 💡 Common Pitfalls

1. **Learning Rate Too High**: May overshoot optimal weights
2. **Learning Rate Too Low**: Very slow convergence
3. **Not Linearly Separable**: Will never converge (like XOR)
4. **Initialization**: Starting weights matter (usually zeros or small random)
5. **Feature Scaling**: May help convergence (though not always necessary)

---

## 🚀 Next Steps

After understanding perceptrons, you'll learn:
- **Multi-Layer Perceptrons (MLPs)**: Multiple layers with hidden neurons
- **Non-linear Activation Functions**: ReLU, Sigmoid, Tanh
- **Backpropagation**: How to train multi-layer networks
- **Universal Approximation**: MLPs can approximate any function

