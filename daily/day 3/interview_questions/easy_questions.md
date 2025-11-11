# Day 3 - Easy Interview Questions

## 1. What is a Multi-Layer Perceptron (MLP)?

**Answer:**
A Multi-Layer Perceptron (MLP) is a type of feedforward neural network with at least three layers:
1. **Input Layer**: Receives the input features (passive layer, no computation)
2. **Hidden Layer(s)**: One or more layers of neurons between input and output that perform computations
3. **Output Layer**: Produces the final prediction

MLPs extend single perceptrons by stacking multiple layers, allowing the network to learn non-linear relationships and solve problems that single perceptrons cannot (like XOR).

---

## 2. What are the three main layers in an MLP architecture?

**Answer:**
1. **Input Layer**: Holds the input data/features. If you have 8 features, this layer has 8 nodes. This layer is passive and doesn't perform any computation.
2. **Hidden Layer(s)**: The "magic" layer(s) that perform computations. Each neuron in a hidden layer is fully connected to all neurons in the previous layer. You can have one or many hidden layers.
3. **Output Layer**: Produces the final prediction. For binary classification, it might be a single neuron; for multi-class, it might have multiple neurons.

---

## 3. Why do we need MLPs when we already have perceptrons?

**Answer:**
Single perceptrons have a critical limitation: they can only solve **linearly separable** problems (problems where a single straight line can separate the classes). 

MLPs overcome this by:
- Using **hidden layers** that allow the network to learn multiple decision boundaries simultaneously
- Applying **non-linear activation functions** that enable the network to learn complex, curved decision boundaries
- This allows MLPs to solve problems like XOR that single perceptrons cannot

---

## 4. What is forward propagation in an MLP?

**Answer:**
Forward propagation is the process of passing input data **forward** through the network, layer by layer, to compute the final output. 

The process:
1. Input data enters the input layer
2. Data flows through hidden layer(s), where each layer computes: `Z = W @ X + b`, then applies activation: `A = activation(Z)`
3. The activated output from one layer becomes the input to the next layer
4. Finally, the output layer produces the prediction

It's called "forward" because data flows in one direction: input → hidden → output.

---

## 5. What is the purpose of hidden layers in an MLP?

**Answer:**
Hidden layers are the "magic" of MLPs because they:
- Allow the network to learn **multiple decision boundaries** simultaneously (each neuron can learn a different boundary)
- Enable **non-linear transformations** of the input data
- Create **hierarchical feature representations** where each layer learns increasingly complex patterns
- Make it possible to solve **non-linearly separable** problems (like XOR)

Without hidden layers, an MLP would just be a single perceptron with limited capabilities.

---

## 6. Why are non-linear activation functions important in MLPs?

**Answer:**
Non-linear activation functions (like Sigmoid or ReLU) are crucial because:
- **Without them**: Stacking multiple layers would just be a series of linear calculations, which could be simplified back to a single linear calculation. You'd gain no additional power.
- **With them**: The activation function "bends" the space, allowing the network to learn complex, curved decision boundaries instead of just straight lines
- They enable the network to model **non-linear relationships** between features
- They are what make the Universal Approximation Theorem possible

---

## 7. What is the Universal Approximation Theorem?

**Answer:**
The Universal Approximation Theorem states that an MLP with a **single hidden layer** (containing a sufficient number of neurons) and a **non-linear activation function** can approximate any continuous function to arbitrary accuracy.

In simple terms: an MLP can learn to draw **any shape** (not just a line), making it capable of solving a wide variety of complex problems, given enough neurons and proper training.

---

## 8. What is the mathematical notation for forward propagation in a 2-layer MLP?

**Answer:**
For a 2-layer MLP (1 hidden layer + 1 output layer):

**Step 1: Input to Hidden Layer**
- $Z^{[1]} = W^{[1]} \cdot X + b^{[1]}$
- $A^{[1]} = \sigma(Z^{[1]})$ (where $\sigma$ is the activation function)

**Step 2: Hidden to Output Layer**
- $Z^{[2]} = W^{[2]} \cdot A^{[1]} + b^{[2]}$
- $A^{[2]} = \sigma(Z^{[2]})$ (final output)

**Notation:**
- $W^{[1]}, b^{[1]}$: weights and bias for first layer (input → hidden)
- $W^{[2]}, b^{[2]}$: weights and bias for second layer (hidden → output)
- $Z$: weighted sum (logit)
- $A$: activated output

---

## 9. What are the typical shapes of matrices in an MLP forward pass?

**Answer:**
Following the convention:
- **Input Data (X)**: `(n_features, n_samples)` - e.g., `(2, 100)` for 2 features and 100 samples
- **Weights (W)**: `(output_neurons, input_neurons)` - e.g., `W^{[1]}` shape `(4, 2)` means 4 hidden neurons receiving 2 inputs
- **Bias (b)**: `(output_neurons, 1)` - e.g., `(4, 1)` for 4 hidden neurons
- **Z (weighted sum)**: `(output_neurons, n_samples)` - e.g., `(4, 100)` for 4 hidden neurons and 100 samples
- **A (activated output)**: Same shape as Z

---

## 10. Can an MLP solve the XOR problem? Why or why not?

**Answer:**
**Yes!** An MLP can solve the XOR problem, which a single perceptron cannot.

**Why:**
- XOR is **not linearly separable** - you cannot draw a single straight line to separate the classes
- An MLP with a hidden layer can learn **multiple decision boundaries** (one per hidden neuron)
- With non-linear activation functions, these boundaries can be combined to create a **curved or complex decision boundary** that separates XOR classes correctly
- This demonstrates the power of hidden layers and non-linearity

---

## 11. What is the difference between Z and A in forward propagation?

**Answer:**
- **Z (weighted sum/logit)**: The result of the linear transformation: `Z = W @ X + b`. This is just the weighted sum before applying any activation function.
- **A (activated output)**: The result after applying the activation function: `A = activation(Z)`. This is the non-linear transformation of Z.

**Flow**: Input → Z (linear) → A (non-linear) → becomes input to next layer

For example:
- `Z^{[1]} = W^{[1]} @ X + b^{[1]}` (linear)
- `A^{[1]} = sigmoid(Z^{[1]})` (non-linear)
- Then `A^{[1]}` becomes the input for the next layer

---

## 12. What happens to the output of one layer in an MLP?

**Answer:**
The output of one layer (the activated output `A`) becomes the **input** for the next layer.

For example:
- Input layer provides `X` to the first hidden layer
- First hidden layer computes `A^{[1]}` and passes it to the second hidden layer (or output layer)
- This creates a chain: `X → A^{[1]} → A^{[2]} → ... → final output`

This layer-by-layer processing is what allows MLPs to learn hierarchical representations.

---

## 13. Why do we initialize weights to small random values instead of zeros in an MLP?

**Answer:**
Initializing weights to small random values (e.g., `np.random.randn() * 0.01`) is important because:
- **Symmetry Breaking**: If all weights start at zero, all neurons in a layer would compute the same thing (symmetry), making them redundant
- **Gradient Flow**: Small random values help gradients flow properly during backpropagation (training)
- **Learning**: Different initial weights allow different neurons to learn different features/patterns
- **Convergence**: Helps the network converge faster and avoid getting stuck

Zero initialization would make training ineffective because all neurons would update identically.

---

## 14. What is a computational graph in the context of neural networks?

**Answer:**
A computational graph is a visual representation of the flow of data and operations in a neural network. It shows:
- **Nodes**: Represent operations (matrix multiplication, addition, activation functions)
- **Edges**: Represent data flow (inputs, outputs, gradients)
- **Flow**: Shows how data moves from input through layers to output

For an MLP, the graph shows: `X, W1, b1 → matmul & add → Z1 → activation → A1 → W2, b2 → matmul & add → Z2 → activation → Output`

Computational graphs are the blueprint that deep learning frameworks use to calculate outputs (forward pass) and gradients (backward pass).

---

## 15. How many parameters does a simple MLP have?

**Answer:**
For an MLP with:
- Input size: `n_input`
- Hidden layer size: `n_hidden`
- Output size: `n_output`

**Parameters:**
- `W^{[1]}`: `n_hidden × n_input` weights
- `b^{[1]}`: `n_hidden` biases
- `W^{[2]}`: `n_output × n_hidden` weights
- `b^{[2]}`: `n_output` biases

**Total**: `(n_hidden × n_input) + n_hidden + (n_output × n_hidden) + n_output`

**Example**: MLP with 2 inputs, 4 hidden neurons, 1 output:
- `W^{[1]}`: 4 × 2 = 8
- `b^{[1]}`: 4
- `W^{[2]}`: 1 × 4 = 4
- `b^{[2]}`: 1
- **Total: 17 parameters**

---

## 16. What is the difference between a single perceptron and an MLP?

**Answer:**

| Aspect | Single Perceptron | MLP |
|--------|------------------|-----|
| **Layers** | 1 layer (input → output) | 3+ layers (input → hidden → output) |
| **Capability** | Only linearly separable problems | Non-linearly separable problems |
| **Decision Boundary** | Single straight line | Complex, curved boundaries |
| **Activation** | Step function | Non-linear functions (Sigmoid, ReLU, etc.) |
| **Can solve XOR?** | No | Yes |
| **Complexity** | Simple | More complex |

**Key Difference**: MLPs add hidden layers with non-linear activations, enabling them to learn complex patterns that single perceptrons cannot.

---

## 17. What does "fully connected" mean in the context of MLPs?

**Answer:**
"Fully connected" means that **every neuron in one layer is connected to every neuron in the next layer**. 

For example, if you have:
- Input layer with 3 neurons
- Hidden layer with 4 neurons

Then each of the 4 hidden neurons receives input from all 3 input neurons. This creates 3 × 4 = 12 connections (weights) between the input and hidden layers.

This is also called a "dense" layer in modern frameworks like TensorFlow/Keras.

---

## 18. Why can't we just stack multiple linear layers without activation functions?

**Answer:**
If you stack multiple linear layers (matrix multiplications and additions) without activation functions, you're just doing:
- `Layer 1`: `Z1 = W1 @ X + b1`
- `Layer 2`: `Z2 = W2 @ Z1 + b2 = W2 @ (W1 @ X + b1) + b2`

This can be simplified to: `Z2 = (W2 @ W1) @ X + (W2 @ b1 + b2)`, which is equivalent to a **single linear layer**.

**Result**: You gain no additional modeling power - it's still just a linear transformation. You need non-linear activation functions to create non-linear decision boundaries and unlock the power of deep networks.

---

## 19. What is the role of the input layer in an MLP?

**Answer:**
The input layer is a **passive layer** that:
- Holds/receives the input data (features)
- Has one neuron (node) per input feature
- Does **not** perform any computation (no weights, no activation)
- Simply passes the data to the first hidden layer

For example, if your data has 8 features, the input layer has 8 nodes, and these 8 values are passed directly to the hidden layer.

---

## 20. How does an MLP use matrix multiplication for efficiency?

**Answer:**
Instead of processing one sample at a time, MLPs use matrix multiplication to process **all samples simultaneously**:

- **Input X**: Shape `(n_features, n_samples)` - contains all samples at once
- **Forward pass**: `Z = W @ X + b` computes outputs for all samples in one operation
- **Result**: Shape `(n_neurons, n_samples)` - outputs for all samples

**Benefits:**
- **Efficiency**: Matrix operations are highly optimized (can use GPU)
- **Speed**: Much faster than looping through samples
- **Vectorization**: Leverages parallel computation

**Example**: With 100 samples, `W @ X` computes all 100 forward passes at once, rather than 100 separate calculations.

---

