# Day 3 - Medium Interview Questions

## 1. Explain the forward propagation process in an MLP step-by-step with mathematical details. How does data flow from input to output?

**Answer:**

**Forward Propagation Process:**

**Step 1: Input Layer to Hidden Layer**

Given input `X` with shape `(n_features, n_samples)`:
- Compute weighted sum: $Z^{[1]} = W^{[1]} \cdot X + b^{[1]}$
  - $W^{[1]}$ shape: `(n_hidden, n_features)`
  - $X$ shape: `(n_features, n_samples)`
  - $Z^{[1]}$ shape: `(n_hidden, n_samples)`
  - Broadcasting: $b^{[1]}$ shape `(n_hidden, 1)` is added to each column

- Apply activation: $A^{[1]} = \sigma(Z^{[1]})$
  - $\sigma$ is the activation function (e.g., sigmoid, ReLU)
  - $A^{[1]}$ shape: `(n_hidden, n_samples)`

**Step 2: Hidden Layer to Output Layer**

- Compute weighted sum: $Z^{[2]} = W^{[2]} \cdot A^{[1]} + b^{[2]}$
  - $W^{[2]}$ shape: `(n_output, n_hidden)`
  - $A^{[1]}$ shape: `(n_hidden, n_samples)`
  - $Z^{[2]}$ shape: `(n_output, n_samples)`

- Apply activation: $A^{[2]} = \sigma(Z^{[2]})$
  - $A^{[2]}$ shape: `(n_output, n_samples)`
  - This is the final output/prediction

**Complete Flow:**
```
X (input) 
  → Z[1] = W[1] @ X + b[1] (linear transformation)
  → A[1] = σ(Z[1]) (non-linear activation)
  → Z[2] = W[2] @ A[1] + b[2] (linear transformation)
  → A[2] = σ(Z[2]) (output activation)
  → Output
```

**Key Points:**
- Each layer's output becomes the next layer's input
- Linear transformation (matrix multiplication + bias) followed by non-linear activation
- All samples processed simultaneously via matrix operations

---

## 2. Why is non-linearity essential in MLPs? What happens mathematically if we remove activation functions?

**Answer:**

**Why Non-Linearity is Essential:**

1. **Enables Non-Linear Decision Boundaries**
   - Without non-linearity, MLPs can only learn linear boundaries
   - Real-world problems require curved, complex boundaries

2. **Unlocks Universal Approximation**
   - Universal Approximation Theorem requires non-linear activations
   - Allows approximation of any continuous function

3. **Creates Hierarchical Features**
   - Each layer learns increasingly complex features
   - Non-linearity enables feature composition

**Mathematical Analysis Without Activation Functions:**

Consider a 2-layer MLP without activations:
- Layer 1: $Z^{[1]} = W^{[1]} \cdot X + b^{[1]}$
- Layer 2: $Z^{[2]} = W^{[2]} \cdot Z^{[1]} + b^{[2]}$

Substituting:
$$Z^{[2]} = W^{[2]} \cdot (W^{[1]} \cdot X + b^{[1]}) + b^{[2]}$$

Expanding:
$$Z^{[2]} = (W^{[2]} \cdot W^{[1]}) \cdot X + (W^{[2]} \cdot b^{[1]} + b^{[2]})$$

This simplifies to:
$$Z^{[2]} = W_{eq} \cdot X + b_{eq}$$

Where:
- $W_{eq} = W^{[2]} \cdot W^{[1]}$ (equivalent weight matrix)
- $b_{eq} = W^{[2]} \cdot b^{[1]} + b^{[2]}$ (equivalent bias)

**Result**: Multiple linear layers collapse into a **single linear layer**. You gain no additional modeling power - it's still just a linear transformation.

**With Non-Linearity:**
- $A^{[1]} = \sigma(Z^{[1]})$ introduces non-linearity
- The composition $W^{[2]} \cdot \sigma(W^{[1]} \cdot X + b^{[1]}) + b^{[2]}$ cannot be simplified
- This creates non-linear decision boundaries

**Conclusion**: Non-linear activations are what make deep networks powerful. Without them, depth provides no benefit.

---

## 3. Explain the Universal Approximation Theorem. What are its assumptions, implications, and limitations?

**Answer:**

**Theorem Statement:**
An MLP with a single hidden layer containing a sufficient number of neurons and a non-linear activation function can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary accuracy.

**Key Assumptions:**
1. **Single Hidden Layer**: Theorem applies to networks with one hidden layer
2. **Non-Linear Activation**: Activation function must be non-linear and non-polynomial (e.g., sigmoid, tanh, ReLU)
3. **Continuous Function**: Function to approximate must be continuous
4. **Compact Domain**: Input domain must be bounded (closed and bounded set)
5. **Sufficient Neurons**: Hidden layer must have enough neurons (number depends on function complexity)

**Implications:**

**Positive:**
- **Theoretical Guarantee**: MLPs can theoretically solve any continuous function approximation problem
- **Justifies Architecture**: Explains why MLPs are powerful
- **No Architectural Limitation**: Single hidden layer is sufficient (in theory)

**Practical Limitations:**
1. **Number of Neurons**: "Sufficient" may mean exponentially many neurons for complex functions
2. **Training Difficulty**: Finding optimal weights is non-trivial (no guarantee on learning algorithm)
3. **Overfitting Risk**: Large networks may memorize training data
4. **Computational Cost**: Many neurons = many parameters = expensive training
5. **Depth vs Width**: In practice, deeper networks (more layers) often work better than very wide single layers

**Why We Use Deep Networks:**
- While single hidden layer is theoretically sufficient, deeper networks:
  - Learn more efficiently (fewer parameters for same capacity)
  - Learn hierarchical features better
  - Generalize better in practice
  - Are easier to train with modern techniques

**Example:**
- Approximating XOR: Needs at least 2 hidden neurons
- Approximating complex image classification: May need thousands of neurons in single layer, or fewer neurons in many layers (deep network)

---

## 4. How do you verify matrix shapes are correct in an MLP forward pass? Walk through a concrete example.

**Answer:**

**Shape Convention:**
- Input `X`: `(n_features, n_samples)`
- Weights `W`: `(output_neurons, input_neurons)`
- Bias `b`: `(output_neurons, 1)`
- Output `Z`/`A`: `(output_neurons, n_samples)`

**Concrete Example:**

**Setup:**
- Input features: 3
- Training samples: 100
- Hidden layer neurons: 5
- Output neurons: 2

**Step 1: Input to Hidden**

Input `X`:
- Shape: `(3, 100)` ✓ (3 features, 100 samples)

Weights `W^{[1]}`:
- Shape: `(5, 3)` ✓ (5 hidden neurons, 3 input features)

Bias `b^{[1]}`:
- Shape: `(5, 1)` ✓ (5 hidden neurons)

Weighted Sum `Z^{[1]}`:
- Calculation: `W^{[1]} @ X`
- Matrix multiplication: `(5, 3) @ (3, 100)`
- Result shape: `(5, 100)` ✓
- Broadcasting: `b^{[1]}` shape `(5, 1)` broadcasts to `(5, 100)`

Activated Output `A^{[1]}`:
- Shape: `(5, 100)` ✓ (same as `Z^{[1]}`)

**Step 2: Hidden to Output**

Weights `W^{[2]}`:
- Shape: `(2, 5)` ✓ (2 output neurons, 5 hidden neurons)

Bias `b^{[2]}`:
- Shape: `(2, 1)` ✓ (2 output neurons)

Weighted Sum `Z^{[2]}`:
- Calculation: `W^{[2]} @ A^{[1]}`
- Matrix multiplication: `(2, 5) @ (5, 100)`
- Result shape: `(2, 100)` ✓
- Broadcasting: `b^{[2]}` shape `(2, 1)` broadcasts to `(2, 100)`

Final Output `A^{[2]}`:
- Shape: `(2, 100)` ✓ (2 outputs for each of 100 samples)

**Verification Rules:**
1. Inner dimensions must match: `(m, n) @ (n, p)` → `(m, p)`
2. Bias must broadcast correctly: `(m, 1)` broadcasts to `(m, p)`
3. Output shape = `(output_neurons, n_samples)`

**Common Mistakes:**
- Transposing input incorrectly: Should be `(features, samples)`, not `(samples, features)`
- Wrong weight matrix orientation: Should be `(output, input)`, not `(input, output)`
- Forgetting to add bias or adding incorrectly shaped bias

---

## 5. Explain how an MLP solves the XOR problem. Why does it work when a single perceptron cannot?

**Answer:**

**XOR Problem Recap:**
- (0, 0) → 0
- (0, 1) → 1
- (1, 0) → 1
- (1, 1) → 0

**Why Single Perceptron Fails:**
- XOR is not linearly separable
- No single straight line can separate the classes
- Points form an "X" pattern: (0,1) and (1,0) are class 1, while (0,0) and (1,1) are class 0

**How MLP Solves It:**

**Architecture:**
- Input: 2 features
- Hidden layer: 2 neurons (minimum needed)
- Output: 1 neuron

**Strategy:**
The hidden layer creates **two intermediate decision boundaries**, and the output layer combines them.

**Hidden Layer Neurons:**
- **Neuron 1**: Learns to detect when inputs are different (XOR-like behavior)
  - High activation when: (0,1) or (1,0)
  - Low activation when: (0,0) or (1,1)

- **Neuron 2**: Can learn complementary patterns or act as a threshold adjuster

**Output Layer:**
- Combines the hidden layer outputs
- With proper weights, it can create a decision boundary that separates:
  - Low activation from both hidden neurons → output 0
  - High activation from at least one hidden neuron → output 1

**Mathematical Example (with pre-set weights):**

Using sigmoid activation and weights:
- `W1 = [[20, 20], [-20, -20]]`
- `b1 = [[-10], [30]]`
- `W2 = [[20, 20]]`
- `b2 = [[-30]]`

**For input (0, 1):**
- Hidden layer: High activation from first neuron, low from second
- Output: Combines to produce ~1

**For input (1, 1):**
- Hidden layer: Both neurons have similar activations
- Output: Combines to produce ~0

**Key Insight:**
- Hidden layer creates **non-linear feature transformations**
- Each hidden neuron learns a different "line" or decision boundary
- Output layer combines these boundaries to create a **curved/complex boundary**
- Non-linear activation (sigmoid) enables this combination

**Why It Works:**
1. **Multiple Boundaries**: Hidden layer provides multiple decision boundaries
2. **Non-Linearity**: Activation functions allow non-linear combination
3. **Feature Learning**: Hidden layer learns intermediate features (e.g., "inputs are different")
4. **Composition**: Output layer composes these features into final decision

**Generalization:**
This demonstrates that MLPs can learn **any** non-linearly separable pattern by:
- Using hidden layers to create multiple boundaries
- Combining them non-linearly
- Learning hierarchical feature representations

---

## 6. What is a computational graph? How does it relate to forward and backward propagation?

**Answer:**

**Definition:**
A computational graph is a directed acyclic graph (DAG) that represents the flow of data and operations in a neural network. It's the blueprint that deep learning frameworks use for automatic differentiation.

**Components:**
- **Nodes**: Represent operations (matrix multiplication, addition, activation functions) or data (inputs, weights, biases)
- **Edges**: Represent data flow (forward: inputs/outputs, backward: gradients)
- **Direction**: Forward edges show data flow, backward edges show gradient flow

**Example for 2-Layer MLP:**

```
Forward Pass:
X ──┐
    ├─→ [MatMul] ─→ Z1 ─→ [Add b1] ─→ Z1' ─→ [Sigmoid] ─→ A1 ──┐
W1 ─┘                                                           │
b1 ────────────────────────────────────────────────────────────┘
                                                                 │
                                                                 ├─→ [MatMul] ─→ Z2 ─→ [Add b2] ─→ Z2' ─→ [Sigmoid] ─→ A2 (Output)
W2 ─────────────────────────────────────────────────────────────┘
b2 ─────────────────────────────────────────────────────────────┘
```

**Detailed Operations:**
1. `MatMul(W1, X)` → `Z1_temp`
2. `Add(Z1_temp, b1)` → `Z1`
3. `Sigmoid(Z1)` → `A1`
4. `MatMul(W2, A1)` → `Z2_temp`
5. `Add(Z2_temp, b2)` → `Z2`
6. `Sigmoid(Z2)` → `A2`

**Relation to Forward Propagation:**
- Forward propagation follows the graph from input to output
- Each node computes its output based on incoming edges
- Results are stored for later use in backpropagation

**Relation to Backward Propagation:**
- Backpropagation follows the graph **backwards** (reverse direction)
- Each node receives gradients from downstream nodes
- Computes local gradients using chain rule
- Passes gradients to upstream nodes

**Backward Pass (Gradient Flow):**
```
dA2 ─→ [dSigmoid] ─→ dZ2 ─→ [dAdd] ─→ dZ2_temp ─→ [dMatMul] ─→ dA1, dW2
                                                                  │
                                                                  └─→ dA1 ─→ [dSigmoid] ─→ dZ1 ─→ [dAdd] ─→ dZ1_temp ─→ [dMatMul] ─→ dX, dW1
```

**Benefits:**
1. **Automatic Differentiation**: Framework automatically computes gradients
2. **Optimization**: Can optimize computation order
3. **Memory Management**: Can decide what to store/forget
4. **Visualization**: Helps understand network structure
5. **Debugging**: Can trace data flow and identify issues

**Implementation in Frameworks:**
- **PyTorch**: Dynamic graphs (built during forward pass)
- **TensorFlow**: Static graphs (defined before execution) or eager execution
- Both use computational graphs under the hood for backpropagation

---

## 7. How do you implement forward propagation efficiently using NumPy? What are common pitfalls?

**Answer:**

**Efficient Implementation:**

```python
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        
        # Initialize biases to zero
        self.b1 = np.zeros((hidden_size, 1))
        self.b2 = np.zeros((output_size, 1))
    
    def _sigmoid(self, z):
        # Numerically stable sigmoid
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, X):
        """
        X shape: (input_size, n_samples)
        """
        # Step 1: Input to Hidden
        Z1 = np.matmul(self.W1, X) + self.b1  # or: self.W1 @ X + self.b1
        A1 = self._sigmoid(Z1)
        
        # Step 2: Hidden to Output
        Z2 = np.matmul(self.W2, A1) + self.b2
        A2 = self._sigmoid(Z2)
        
        return A2
```

**Efficiency Techniques:**

1. **Matrix Operations**: Use `np.matmul()` or `@` operator instead of loops
2. **Broadcasting**: Let NumPy handle bias broadcasting automatically
3. **Vectorization**: Process all samples simultaneously

**Common Pitfalls:**

1. **Wrong Matrix Shapes**
   ```python
   # WRONG: Transposed input
   X = X.T  # Should be (features, samples), not (samples, features)
   
   # CORRECT
   X = X  # Shape: (features, samples)
   ```

2. **Incorrect Weight Matrix Orientation**
   ```python
   # WRONG
   W1 = np.random.randn(input_size, hidden_size)  # Wrong orientation
   
   # CORRECT
   W1 = np.random.randn(hidden_size, input_size)  # (output, input)
   ```

3. **Broadcasting Issues**
   ```python
   # WRONG: Bias shape mismatch
   b1 = np.zeros(hidden_size)  # 1D array, won't broadcast correctly
   
   # CORRECT
   b1 = np.zeros((hidden_size, 1))  # 2D column vector
   ```

4. **Numerical Instability**
   ```python
   # WRONG: Can overflow for large z
   def sigmoid(z):
       return 1 / (1 + np.exp(-z))
   
   # CORRECT: Clip to prevent overflow
   def sigmoid(z):
       return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
   ```

5. **Using Wrong Multiplication**
   ```python
   # WRONG: Element-wise multiplication
   Z1 = self.W1 * X + self.b1  # * is element-wise, not matrix multiplication
   
   # CORRECT: Matrix multiplication
   Z1 = np.matmul(self.W1, X) + self.b1  # or: self.W1 @ X + self.b1
   ```

6. **Not Storing Intermediate Values**
   ```python
   # WRONG: Can't use in backpropagation
   return self._sigmoid(self.W2 @ self._sigmoid(self.W1 @ X + self.b1) + self.b2)
   
   # CORRECT: Store intermediate values
   Z1 = self.W1 @ X + self.b1
   A1 = self._sigmoid(Z1)
   Z2 = self.W2 @ A1 + self.b2
   A2 = self._sigmoid(Z2)
   return A2, (Z1, A1, Z2)  # Return for backprop
   ```

7. **Memory Inefficiency**
   ```python
   # WRONG: Creating unnecessary copies
   temp = X.copy()
   Z1 = self.W1 @ temp + self.b1
   
   # CORRECT: Use views when possible
   Z1 = self.W1 @ X + self.b1  # X unchanged, no copy needed
   ```

**Best Practices:**
- Always verify shapes: `print(f"Shape: {X.shape}")`
- Use `np.matmul()` or `@` for matrix multiplication
- Initialize biases as column vectors: `(n, 1)` not `(n,)`
- Store intermediate values for backpropagation
- Use numerically stable activation functions
- Test with known inputs/outputs

---

## 8. How does the number of hidden neurons affect an MLP's capacity? What are the trade-offs?

**Answer:**

**Capacity Definition:**
Capacity refers to the model's ability to represent complex functions. More neurons = higher capacity = can learn more complex patterns.

**Effect of Hidden Neurons:**

**Too Few Neurons (Underfitting):**
- **Problem**: Insufficient capacity to learn the underlying pattern
- **Symptoms**:
  - High training error
  - High validation error
  - Model too simple for the data
- **Example**: 1 hidden neuron for XOR problem (cannot solve)
- **Solution**: Increase number of neurons

**Optimal Number:**
- **Balance**: Enough neurons to learn the pattern, but not so many as to overfit
- **Rule of Thumb**: Start with hidden neurons = input features to 2× input features
- **XOR Example**: Minimum 2 hidden neurons needed

**Too Many Neurons (Overfitting Risk):**
- **Problem**: Excessive capacity can memorize training data
- **Symptoms**:
  - Very low training error
  - High validation error (poor generalization)
  - Model too complex, learns noise
- **Solution**: Regularization, dropout, or reduce neurons

**Mathematical Relationship:**

**Parameters vs Neurons:**
For MLP with `n_input` inputs, `n_hidden` hidden neurons, `n_output` outputs:
- Total parameters: `(n_hidden × n_input) + n_hidden + (n_output × n_hidden) + n_output`
- Approximately: `O(n_hidden × (n_input + n_output))`

**Capacity Growth:**
- Each hidden neuron adds a decision boundary/hyperplane
- More neurons = more boundaries = more complex decision regions
- But: More parameters = harder to train, more data needed

**Trade-offs:**

| Aspect | Few Neurons | Many Neurons |
|--------|-------------|--------------|
| **Capacity** | Low | High |
| **Parameters** | Few | Many |
| **Training Speed** | Fast | Slow |
| **Memory** | Low | High |
| **Overfitting Risk** | Low | High |
| **Generalization** | May underfit | May overfit |
| **Data Requirements** | Less data needed | More data needed |

**Practical Guidelines:**

1. **Start Small**: Begin with fewer neurons, increase if underfitting
2. **Use Validation Set**: Monitor validation error to detect overfitting
3. **Consider Data Size**: More data allows more neurons without overfitting
4. **Regularization**: Use dropout/L2 regularization to allow more neurons safely
5. **Architecture Search**: Try different numbers: 32, 64, 128, 256, etc.

**Example Progression:**
```python
# Too few (may underfit)
mlp1 = MLP(input_size=10, hidden_size=2, output_size=1)

# Moderate
mlp2 = MLP(input_size=10, hidden_size=16, output_size=1)

# Many (may overfit without regularization)
mlp3 = MLP(input_size=10, hidden_size=256, output_size=1)
```

**Depth vs Width:**
- **Wide (many neurons, few layers)**: Can work but inefficient
- **Deep (fewer neurons, more layers)**: Often more efficient, learns hierarchical features better
- Modern approach: Prefer depth over extreme width

---

## 9. Explain the difference between the input layer, hidden layer, and output layer in terms of computation and purpose.

**Answer:**

**Input Layer:**

**Purpose:**
- Receives and holds input data
- No computation performed
- Simply passes data to first hidden layer

**Characteristics:**
- **Passive**: No weights, no biases, no activation
- **Size**: One neuron per input feature
- **Output**: Raw input data (no transformation)
- **Role**: Data holder/interface

**Example:**
- 8 features → 8 input neurons
- Input: `[x1, x2, ..., x8]` → Output: `[x1, x2, ..., x8]` (unchanged)

**Hidden Layer(s):**

**Purpose:**
- Perform **non-linear transformations** of input data
- Learn **intermediate features** and representations
- Create **multiple decision boundaries**
- Enable non-linear modeling capability

**Characteristics:**
- **Active**: Has weights, biases, and activation functions
- **Size**: Configurable (design choice)
- **Computation**: 
  - Linear: `Z = W @ X + b`
  - Non-linear: `A = activation(Z)`
- **Role**: Feature learning and transformation

**What It Learns:**
- First hidden layer: Low-level features (edges, patterns)
- Deeper hidden layers: High-level features (combinations of lower-level features)
- Each neuron learns a different feature/pattern

**Example:**
- Input: `[x1, x2]` (2 features)
- Hidden layer (4 neurons):
  - Neuron 1: Learns pattern like "x1 > x2"
  - Neuron 2: Learns pattern like "x1 + x2 > threshold"
  - Neuron 3: Learns another pattern
  - Neuron 4: Learns another pattern
- Output: `[a1, a2, a3, a4]` (4 transformed features)

**Output Layer:**

**Purpose:**
- Produce **final prediction** or output
- Map learned features to target output format
- Apply task-specific activation (sigmoid for binary, softmax for multi-class)

**Characteristics:**
- **Active**: Has weights, biases, and activation functions
- **Size**: Depends on task
  - Binary classification: 1 neuron
  - Multi-class: K neurons (one per class)
  - Regression: 1 neuron (no activation or linear)
- **Computation**: 
  - Linear: `Z = W @ A_prev + b`
  - Activation: `Output = activation(Z)` (task-specific)
- **Role**: Final decision maker

**Activation Functions:**
- Binary classification: Sigmoid (output: probability)
- Multi-class: Softmax (output: probability distribution)
- Regression: None or linear (output: continuous value)

**Example:**
- Input from hidden layer: `[a1, a2, a3, a4]` (4 features)
- Output layer (1 neuron for binary classification):
  - Computes: `z = w1*a1 + w2*a2 + w3*a3 + w4*a4 + b`
  - Applies sigmoid: `output = σ(z)` (probability between 0 and 1)

**Summary Table:**

| Layer | Computation | Weights | Activation | Purpose |
|-------|-------------|---------|------------|---------|
| **Input** | None (passive) | No | No | Hold input data |
| **Hidden** | `Z = W@X+b`, `A = σ(Z)` | Yes | Yes (non-linear) | Learn features, transform data |
| **Output** | `Z = W@A+b`, `Output = f(Z)` | Yes | Yes (task-specific) | Produce final prediction |

**Key Insight:**
- Input layer: **No learning** (just data)
- Hidden layers: **Feature learning** (the "magic")
- Output layer: **Decision making** (final prediction)

---

## 10. How would you debug shape mismatches in an MLP forward pass? Provide a systematic approach.

**Answer:**

**Systematic Debugging Approach:**

**Step 1: Print Shapes at Each Step**

```python
def forward_debug(self, X):
    print(f"Input X shape: {X.shape}")
    
    # Step 1: Input to Hidden
    print(f"W1 shape: {self.W1.shape}")
    print(f"b1 shape: {self.b1.shape}")
    
    Z1 = np.matmul(self.W1, X) + self.b1
    print(f"Z1 shape after matmul: {np.matmul(self.W1, X).shape}")
    print(f"Z1 shape after adding b1: {Z1.shape}")
    
    A1 = self._sigmoid(Z1)
    print(f"A1 shape: {A1.shape}")
    
    # Step 2: Hidden to Output
    print(f"W2 shape: {self.W2.shape}")
    print(f"b2 shape: {self.b2.shape}")
    
    Z2 = np.matmul(self.W2, A1) + self.b2
    print(f"Z2 shape after matmul: {np.matmul(self.W2, A1).shape}")
    print(f"Z2 shape after adding b2: {Z2.shape}")
    
    A2 = self._sigmoid(Z2)
    print(f"A2 (output) shape: {A2.shape}")
    
    return A2
```

**Step 2: Verify Shape Conventions**

Check that you're following the convention:
- Input: `(n_features, n_samples)`
- Weights: `(output_neurons, input_neurons)`
- Bias: `(output_neurons, 1)`

**Step 3: Matrix Multiplication Rules**

For `A @ B` to work:
- `A` shape: `(m, n)`
- `B` shape: `(n, p)`
- Result: `(m, p)`

**Inner dimensions must match!**

**Step 4: Common Shape Errors and Fixes**

**Error 1: Input Transposed**
```python
# Problem
X = X.T  # Shape: (n_samples, n_features) - WRONG

# Fix
X = X  # Shape: (n_features, n_samples) - CORRECT
# Or if data comes as (samples, features):
X = X.T  # Transpose to (features, samples)
```

**Error 2: Weight Matrix Wrong Orientation**
```python
# Problem
W1 = np.random.randn(input_size, hidden_size)  # WRONG

# Fix
W1 = np.random.randn(hidden_size, input_size)  # CORRECT
# Rule: (output_neurons, input_neurons)
```

**Error 3: Bias Not Column Vector**
```python
# Problem
b1 = np.zeros(hidden_size)  # Shape: (hidden_size,) - WRONG, 1D

# Fix
b1 = np.zeros((hidden_size, 1))  # Shape: (hidden_size, 1) - CORRECT, 2D
```

**Error 4: Using Element-wise Multiplication**
```python
# Problem
Z1 = self.W1 * X + self.b1  # * is element-wise, shapes must match exactly

# Fix
Z1 = np.matmul(self.W1, X) + self.b1  # Matrix multiplication
# Or: Z1 = self.W1 @ X + self.b1
```

**Step 5: Shape Verification Function**

```python
def verify_shapes(self, X):
    """Verify all shapes are correct before forward pass"""
    n_features, n_samples = X.shape
    
    # Check W1
    assert self.W1.shape == (self.hidden_size, n_features), \
        f"W1 shape mismatch: expected ({self.hidden_size}, {n_features}), got {self.W1.shape}"
    
    # Check b1
    assert self.b1.shape == (self.hidden_size, 1), \
        f"b1 shape mismatch: expected ({self.hidden_size}, 1), got {self.b1.shape}"
    
    # Check W2
    assert self.W2.shape == (self.output_size, self.hidden_size), \
        f"W2 shape mismatch: expected ({self.output_size}, {self.hidden_size}), got {self.W2.shape}"
    
    # Check b2
    assert self.b2.shape == (self.output_size, 1), \
        f"b2 shape mismatch: expected ({self.output_size}, 1), got {self.b2.shape}"
    
    print("✓ All shapes verified!")
```

**Step 6: Test with Known Dimensions**

```python
# Test with simple, known dimensions
mlp = MLP(input_size=2, hidden_size=3, output_size=1)
X_test = np.random.randn(2, 5)  # 2 features, 5 samples

print("Expected shapes:")
print(f"X: (2, 5)")
print(f"W1: (3, 2)")
print(f"b1: (3, 1)")
print(f"Z1/A1: (3, 5)")
print(f"W2: (1, 3)")
print(f"b2: (1, 1)")
print(f"Z2/A2: (1, 5)")

output = mlp.forward_debug(X_test)
```

**Step 7: Use Assertions**

```python
def forward(self, X):
    # Assert input shape
    assert X.ndim == 2, f"X must be 2D, got {X.ndim}D"
    assert X.shape[0] == self.input_size, \
        f"X features ({X.shape[0]}) != input_size ({self.input_size})"
    
    Z1 = np.matmul(self.W1, X) + self.b1
    assert Z1.shape == (self.hidden_size, X.shape[1]), \
        f"Z1 shape error: {Z1.shape}"
    
    A1 = self._sigmoid(Z1)
    Z2 = np.matmul(self.W2, A1) + self.b2
    assert Z2.shape == (self.output_size, X.shape[1]), \
        f"Z2 shape error: {Z2.shape}"
    
    A2 = self._sigmoid(Z2)
    return A2
```

**Step 8: Visualize Shape Flow**

Create a diagram:
```
Input: (2, 100)
  ↓
W1 @ X: (3, 2) @ (2, 100) → (3, 100) ✓
  ↓
+ b1: (3, 1) broadcasts to (3, 100) → (3, 100) ✓
  ↓
A1: (3, 100)
  ↓
W2 @ A1: (1, 3) @ (3, 100) → (1, 100) ✓
  ↓
+ b2: (1, 1) broadcasts to (1, 100) → (1, 100) ✓
  ↓
Output: (1, 100) ✓
```

**Best Practices:**
1. Always print shapes during development
2. Use assertions to catch errors early
3. Test with small, known examples first
4. Document expected shapes in comments
5. Use shape verification function before training
6. Keep shape convention consistent throughout codebase

