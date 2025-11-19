# Day 8 - Easy Interview Questions

## 1. What is a DataLoader in PyTorch?

**Answer:**
A DataLoader is a PyTorch utility that handles efficient data loading, batching, shuffling, and parallel data processing during training.

**Key Features:**
- **Batching**: Groups multiple samples together (e.g., 64 images at once)
- **Shuffling**: Randomly reorders data each epoch (for training)
- **Parallel Loading**: Uses multiple worker processes to load data in the background
- **Iteration**: Provides an easy way to loop through batches

**Example:**
```python
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)
```

**Why It's Important:**
- GPUs process batches efficiently (parallel computation)
- Shuffling prevents the model from learning order-dependent patterns
- Parallel loading prevents data loading from being a bottleneck

---

## 2. What is the difference between `model.train()` and `model.eval()`?

**Answer:**
These methods set the model to different modes that affect how certain layers behave:

**`model.train()` (Training Mode):**
- Enables dropout (randomly deactivates neurons)
- Batch normalization uses batch statistics
- Used during training

**`model.eval()` (Evaluation Mode):**
- Disables dropout (uses all neurons)
- Batch normalization uses running statistics (from training)
- Used during validation/testing

**Why It Matters:**
- During evaluation, you want consistent, deterministic behavior
- Dropout should be off so you get the same prediction for the same input
- Batch norm should use learned statistics, not batch statistics

**Example:**
```python
# Training
model.train()
for batch in train_loader:
    outputs = model(inputs)
    loss.backward()

# Evaluation
model.eval()
with torch.no_grad():
    for batch in test_loader:
        outputs = model(inputs)
```

---

## 3. What is `torch.no_grad()` and why do we use it during evaluation?

**Answer:**
`torch.no_grad()` is a context manager that disables gradient computation in PyTorch.

**What It Does:**
- Tells PyTorch not to build computation graphs
- Prevents storing gradients
- Speeds up computation
- Reduces memory usage

**Why Use It During Evaluation:**
1. **No Need for Gradients**: During evaluation, we're not updating weights, so we don't need gradients
2. **Memory Savings**: Gradients can take significant memory (especially for large models)
3. **Speed**: Building computation graphs has overhead; disabling it speeds up inference
4. **Correctness**: Ensures we're not accidentally updating weights during evaluation

**Example:**
```python
model.eval()
with torch.no_grad():  # Disable gradients
    outputs = model(inputs)
    predictions = torch.argmax(outputs, dim=1)
```

**Without `no_grad()`:**
- PyTorch still builds computation graphs
- Wastes memory and computation
- Slower inference

---

## 4. What is the purpose of `optimizer.zero_grad()` in the training loop?

**Answer:**
`optimizer.zero_grad()` zeros out (resets) the gradients of all model parameters before computing new gradients.

**Why It's Needed:**
- PyTorch **accumulates** gradients by default
- If you don't zero gradients, they add up across batches
- This would cause incorrect weight updates

**What Happens Without It:**
```python
# Batch 1: gradient = 0.5
loss.backward()  # gradients accumulate

# Batch 2: gradient = 0.3
loss.backward()  # gradients become 0.5 + 0.3 = 0.8 (WRONG!)
```

**Correct Usage:**
```python
for batch in train_loader:
    optimizer.zero_grad()  # Reset gradients
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()        # Compute new gradients
    optimizer.step()       # Update weights
```

**Key Point:** Always call `zero_grad()` before `backward()` in each training iteration.

---

## 5. What is the standard training loop pattern in PyTorch?

**Answer:**
The standard training loop follows this pattern:

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # 1. Zero gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        outputs = model(inputs)
        
        # 3. Compute loss
        loss = criterion(outputs, labels)
        
        # 4. Backward pass (compute gradients)
        loss.backward()
        
        # 5. Update weights
        optimizer.step()
```

**Breaking Down Each Step:**

1. **`optimizer.zero_grad()`**: Reset gradients from previous iteration
2. **Forward pass**: Pass data through model to get predictions
3. **Compute loss**: Calculate how wrong predictions are
4. **`loss.backward()`**: Compute gradients using backpropagation
5. **`optimizer.step()`**: Update model weights using gradients

**Why This Order:**
- Must zero gradients before computing new ones
- Must compute loss before backward pass
- Must compute gradients before updating weights

---

## 6. What is the MNIST dataset?

**Answer:**
MNIST (Modified National Institute of Standards and Technology) is a classic dataset of handwritten digits used for image classification.

**Dataset Details:**
- **60,000 training images**
- **10,000 test images**
- **Image size**: 28×28 pixels (grayscale)
- **Classes**: 10 (digits 0-9)
- **Format**: Black and white images of handwritten digits

**Why It's Popular:**
- Simple enough for beginners to understand
- Complex enough to demonstrate real deep learning
- Well-established benchmark
- Fast to train (small images, small dataset)
- Real-world application (digit recognition in banking, postal services)

**Example Use Case:**
Building a model that can look at a handwritten digit image and correctly identify whether it's 0, 1, 2, ..., or 9.

---

## 7. What is data normalization and why do we normalize images?

**Answer:**
Data normalization is the process of scaling data to have a mean of 0 and standard deviation of 1 (or scaling to a specific range).

**For Images:**
- Original pixel values: 0-255 (or 0.0-1.0 after ToTensor)
- Normalized: Mean ≈ 0, Std ≈ 1

**How It's Done:**
```python
transform = transforms.Compose([
    transforms.ToTensor(),  # Scales to [0, 1]
    transforms.Normalize(mean=0.1307, std=0.3081)  # Normalizes
])
```

**Why Normalize:**
1. **Faster Training**: Neural networks learn faster when inputs are centered around 0
2. **Stable Gradients**: Prevents some features (pixels) from dominating others
3. **Better Convergence**: Helps optimization algorithms work better
4. **Consistent Scale**: All features are on the same scale

**Mathematical Effect:**
- Without normalization: Some pixels might have values 0-255, others 0-1
- With normalization: All pixels have similar scale (mean 0, std 1)
- This prevents large values from dominating the learning process

---

## 8. What is a batch size and how do you choose it?

**Answer:**
Batch size is the number of training examples processed together in one forward/backward pass.

**Common Batch Sizes:**
- **32**: Small, good for limited memory
- **64**: Common default, good balance
- **128**: Larger, requires more GPU memory
- **256**: Very large, needs significant memory

**How to Choose:**
1. **GPU Memory**: Larger batches need more memory
2. **Model Size**: Larger models can fit fewer samples per batch
3. **Dataset Size**: Very large datasets might use larger batches
4. **Training Stability**: Larger batches = more stable gradients, but may get stuck in local minima

**Trade-offs:**

**Small Batch (32):**
- ✅ Less GPU memory needed
- ✅ More gradient noise (can help escape local minima)
- ❌ Slower training (more iterations)
- ❌ Less stable gradients

**Large Batch (128-256):**
- ✅ Faster training (fewer iterations)
- ✅ More stable gradients
- ❌ More GPU memory needed
- ❌ May get stuck in poor local minima

**Rule of Thumb:** Start with 64, adjust based on memory and performance.

---

## 9. What is CrossEntropyLoss and when do we use it?

**Answer:**
CrossEntropyLoss is a loss function used for **multi-class classification** problems.

**What It Does:**
- Takes raw logits (unnormalized scores) from the model
- Applies softmax internally to convert to probabilities
- Computes negative log-likelihood loss
- Perfect for problems with multiple classes (e.g., 10 digit classes in MNIST)

**When to Use:**
- Multi-class classification (3+ classes)
- Output layer has no activation (raw logits)
- Each sample belongs to exactly one class

**Example:**
```python
# Model outputs 10 logits (one per class)
outputs = model(images)  # Shape: [batch_size, 10]

# Labels are class indices (0-9)
labels = torch.tensor([3, 7, 1, ...])  # Shape: [batch_size]

# Compute loss
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
```

**Why Not Use MSE for Classification:**
- MSE doesn't understand probabilities
- CrossEntropyLoss is designed for classification
- Better gradients for learning

---

## 10. What does `.to(device)` do in PyTorch?

**Answer:**
`.to(device)` moves tensors or models to a specific computing device (CPU or GPU).

**Common Usage:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleNetwork().to(device)  # Move model to GPU/CPU
images = images.to(device)          # Move data to GPU/CPU
```

**Why It's Important:**
1. **GPU Acceleration**: Training on GPU is 10-100x faster than CPU
2. **Device Consistency**: Model and data must be on the same device
3. **Memory Management**: GPUs have their own memory separate from CPU

**What Happens:**
- Model parameters are moved to the device
- All computations happen on that device
- Data must also be moved to the same device

**Common Error:**
```python
model = model.to('cuda')
images = images  # Still on CPU - ERROR!
# RuntimeError: Expected all tensors to be on the same device
```

**Correct:**
```python
model = model.to('cuda')
images = images.to('cuda')  # Move data too!
```

---

## 11. What is the difference between training loss and test loss?

**Answer:**
Training loss and test loss measure model performance on different datasets:

**Training Loss:**
- Computed on the **training dataset** (data the model learns from)
- Measures how well the model fits the training data
- Used to update model weights during training

**Test Loss:**
- Computed on the **test dataset** (unseen data)
- Measures how well the model generalizes to new data
- Used to evaluate final model performance

**Why Both Matter:**
- **Low training loss, high test loss**: Overfitting (model memorized training data)
- **High training loss, high test loss**: Underfitting (model too simple)
- **Low training loss, low test loss**: Good generalization (ideal!)

**Example:**
- Training loss: 0.1 (model fits training data well)
- Test loss: 0.5 (model struggles on new data)
- **Problem**: Overfitting - model doesn't generalize

---

## 12. What is an epoch in deep learning?

**Answer:**
An epoch is one complete pass through the entire training dataset.

**Example:**
- Training dataset: 60,000 images
- Batch size: 64
- Number of batches per epoch: 60,000 / 64 = 938 batches
- **One epoch** = processing all 938 batches once

**Why Epochs Matter:**
- Training for multiple epochs gives the model multiple chances to learn
- Each epoch sees all training data in (potentially) different order
- More epochs = more learning, but risk of overfitting

**Typical Training:**
```python
for epoch in range(num_epochs):  # e.g., 10 epochs
    for batch in train_loader:   # Process all batches
        train_one_batch()
```

**How Many Epochs:**
- Depends on dataset size and complexity
- Small dataset: May need many epochs (50-100)
- Large dataset: Fewer epochs needed (1-10)
- Monitor validation loss to stop when it stops improving

---

## 13. What is the purpose of shuffling in DataLoader?

**Answer:**
Shuffling randomly reorders the training data each epoch.

**How It Works:**
```python
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True  # Randomly reorder each epoch
)
```

**Why It's Important:**
1. **Prevents Order Bias**: Model doesn't learn patterns based on data order
2. **Better Learning**: Each epoch sees data in different order
3. **Breaks Correlations**: Prevents model from learning spurious sequential patterns

**Example Problem Without Shuffling:**
- If all "0" digits come first, then all "1" digits, etc.
- Model might learn: "First batches = digit 0"
- This is wrong! Order shouldn't matter.

**With Shuffling:**
- Each epoch: Data in random order
- Model learns: "This image = digit X" (not based on position)
- Better generalization

**Note:** Test data usually doesn't need shuffling (order doesn't matter for evaluation).

---

## 14. What is the purpose of flattening in neural networks?

**Answer:**
Flattening converts multi-dimensional data (like images) into a 1D vector for fully connected layers.

**Example:**
```python
# MNIST image: [1, 28, 28] (1 channel, 28x28 pixels)
x = x.view(-1, 28 * 28)  # Flatten to [batch_size, 784]
# Now: [batch_size, 784] - ready for Linear layer
```

**Why It's Needed:**
- Fully connected layers (`nn.Linear`) expect 1D input
- Images are 2D or 3D (height × width × channels)
- Flattening converts 2D/3D → 1D

**Mathematical Effect:**
- Before: Image is 28×28 = 784 pixels arranged in 2D
- After: Same 784 pixels, but in a 1D vector
- Information is preserved, just reshaped

**In Code:**
```python
class SimpleNetwork(nn.Module):
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten: [B, 1, 28, 28] → [B, 784]
        x = self.fc1(x)           # Now can use Linear layer
        return x
```

**Alternative:** `nn.Flatten()` layer does the same thing:
```python
self.flatten = nn.Flatten()  # Automatically flattens
```

---

## 15. What is the difference between accuracy and loss?

**Answer:**
Loss and accuracy are two different ways to measure model performance:

**Loss:**
- **Continuous value**: Measures how "wrong" predictions are
- **Lower is better**: We want to minimize loss
- **Used for training**: Optimizer uses loss gradients to update weights
- **Examples**: CrossEntropyLoss, MSELoss

**Accuracy:**
- **Percentage**: Measures how many predictions are correct
- **Higher is better**: We want to maximize accuracy
- **Used for evaluation**: Easy to interpret (e.g., "95% accurate")
- **Calculation**: (Correct predictions) / (Total predictions) × 100%

**Example:**
```python
# Model predictions
outputs = model(images)
loss = criterion(outputs, labels)  # Loss: 0.234

# Calculate accuracy
_, predicted = torch.max(outputs, 1)
correct = (predicted == labels).sum().item()
accuracy = 100 * correct / len(labels)  # Accuracy: 94.5%
```

**Key Differences:**
- **Loss**: Continuous, used for optimization
- **Accuracy**: Discrete, used for interpretation
- **Relationship**: Lower loss usually means higher accuracy, but not always

**Why Both:**
- Loss: Tells optimizer how to improve
- Accuracy: Tells humans how well model performs

---

## 16. What happens if you forget to call `model.eval()` during evaluation?

**Answer:**
If you forget `model.eval()`, the model stays in training mode, which can cause incorrect evaluation results.

**Problems:**
1. **Dropout Still Active**: Randomly deactivates neurons, making predictions inconsistent
2. **Batch Norm Uses Batch Stats**: Uses current batch statistics instead of learned running statistics
3. **Non-Deterministic**: Same input might give different outputs

**Example:**
```python
# WRONG - Forgot model.eval()
model.train()  # Still in training mode
with torch.no_grad():
    outputs = model(test_images)  # Dropout is still on!
    # Predictions are random and inconsistent
```

**Correct:**
```python
# CORRECT
model.eval()  # Switch to evaluation mode
with torch.no_grad():
    outputs = model(test_images)  # Dropout off, consistent predictions
```

**Impact:**
- Evaluation metrics (accuracy, loss) will be wrong
- Model might appear worse than it actually is
- Predictions are inconsistent (same input → different output)

**Rule:** Always call `model.eval()` before evaluation!

---

## 17. What is the purpose of `transforms.ToTensor()`?

**Answer:**
`transforms.ToTensor()` converts PIL images or numpy arrays to PyTorch tensors and scales pixel values.

**What It Does:**
1. **Converts to Tensor**: PIL Image or numpy array → PyTorch tensor
2. **Scales Values**: Pixel values from [0, 255] → [0.0, 1.0]
3. **Changes Shape**: (H, W) or (H, W, C) → (C, H, W) (channels first)

**Example:**
```python
# Before: PIL Image, values 0-255, shape (28, 28)
# After: Tensor, values 0.0-1.0, shape (1, 28, 28)
transform = transforms.ToTensor()
image_tensor = transform(pil_image)
```

**Why It's Needed:**
- PyTorch models expect tensors, not PIL images
- Neural networks work better with normalized values [0, 1]
- PyTorch convention: channels first (C, H, W) not channels last (H, W, C)

**Common Pipeline:**
```python
transform = transforms.Compose([
    transforms.ToTensor(),      # Convert and scale to [0, 1]
    transforms.Normalize(...)   # Then normalize to mean 0, std 1
])
```

---

## 18. What does `num_workers` do in DataLoader?

**Answer:**
`num_workers` specifies how many parallel processes to use for loading data.

**How It Works:**
```python
train_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=2  # Use 2 parallel processes
)
```

**What It Does:**
- Creates multiple worker processes
- Each worker loads data in parallel
- Prevents data loading from being a bottleneck
- While GPU trains on batch N, workers load batch N+1, N+2, etc.

**Common Values:**
- **0**: No parallel loading (single process, slower)
- **2-4**: Good for most cases
- **8+**: For powerful CPUs, can speed up loading

**Trade-offs:**

**num_workers=0:**
- ✅ Simple, works everywhere
- ❌ Slower (sequential loading)
- ❌ GPU might wait for data

**num_workers=2-4:**
- ✅ Faster data loading
- ✅ GPU doesn't wait
- ❌ Uses more CPU/memory

**Note:** On Windows, sometimes need `num_workers=0` due to multiprocessing issues.

---

## 19. What is the difference between `nn.Linear` and `F.linear`?

**Answer:**
Both perform linear transformations, but `nn.Linear` is a layer (class) while `F.linear` is a function.

**`nn.Linear` (Layer/Module):**
- A class that you instantiate and add to your model
- Stores weights and biases as parameters
- Can be saved/loaded with the model
- Used in `__init__` when defining model architecture

**`F.linear` (Function):**
- A functional interface (no class)
- Takes weights and input as arguments
- More flexible, but you manage weights yourself
- Used in `forward()` for custom operations

**Example:**
```python
# Using nn.Linear (common)
class Model(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(784, 128)  # Layer
    
    def forward(self, x):
        return self.fc(x)  # Use the layer

# Using F.linear (less common)
class Model(nn.Module):
    def __init__(self):
        self.weight = nn.Parameter(torch.randn(128, 784))
        self.bias = nn.Parameter(torch.randn(128))
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)  # Function
```

**When to Use:**
- **nn.Linear**: Standard choice, easier to use
- **F.linear**: When you need more control or custom operations

---

## 20. What is the purpose of storing training history (losses and accuracies)?

**Answer:**
Storing training history allows you to track model performance over time and diagnose training issues.

**What to Store:**
```python
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    test_loss, test_acc = evaluate(...)
    
    # Store for later analysis
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
```

**Why It's Important:**
1. **Visualization**: Plot learning curves to see training progress
2. **Diagnosis**: Identify overfitting, underfitting, or other issues
3. **Comparison**: Compare different models or hyperparameters
4. **Debugging**: Find when problems occurred during training

**What You Can Learn:**
- **Overfitting**: Training loss decreases, test loss increases
- **Underfitting**: Both losses are high and not decreasing
- **Good Training**: Both losses decrease together
- **Learning Rate Issues**: Loss oscillates or diverges

**Example Use:**
```python
# Plot learning curves
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.legend()
plt.show()
```

**Key Point:** Always track training history to understand how your model is learning!

---


