# Day 6 - Easy Interview Questions

## 1. What is a loss function?

**Answer:**
A loss function (also called cost function or objective function) is a mathematical function that measures how far a model's predictions are from the actual target values. It quantifies the "error" or "cost" of the model's predictions.

**Key Purpose:**
- Provides a single number that represents how wrong the model is
- Guides the learning process during training
- Used to compute gradients for backpropagation

**Example:**
- If the true value is 5 and the model predicts 4, the loss should be small
- If the true value is 5 and the model predicts 1, the loss should be large

**Properties:**
- Must be differentiable (for gradient-based optimization)
- Should be appropriate for the problem type (regression vs. classification)
- Lower loss = better predictions

---

## 2. What is Mean Squared Error (MSE)?

**Answer:**
Mean Squared Error (MSE) is a loss function commonly used for **regression problems** (predicting continuous values).

**Formula:**
$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- $y_i$ is the true value
- $\hat{y}_i$ is the predicted value
- $n$ is the number of samples

**Properties:**
- Always positive (squaring ensures this)
- Penalizes large errors more than small errors (quadratic penalty)
- Smooth and differentiable everywhere

**Use Case:**
- Regression problems: predicting house prices, temperatures, stock prices, etc.
- Output layer: Linear activation (no activation) or ReLU

**Example:**
- True values: $[1.0, 2.0, 3.0]$
- Predictions: $[0.9, 2.1, 2.8]$
- MSE: $\frac{(1.0-0.9)^2 + (2.0-2.1)^2 + (3.0-2.8)^2}{3} = \frac{0.01 + 0.01 + 0.04}{3} = 0.02$

---

## 3. What is Binary Cross-Entropy Loss?

**Answer:**
Binary Cross-Entropy (BCE) is a loss function used for **binary classification** problems (two classes: 0 or 1).

**Formula:**
$$L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i)]$$

Where:
- $y_i$ is the true label (0 or 1)
- $\hat{y}_i$ is the predicted probability (between 0 and 1)
- $n$ is the number of samples

**Properties:**
- Works with probabilities (outputs between 0 and 1)
- Measures "surprise" or information content
- Penalizes confident wrong predictions heavily

**Use Case:**
- Binary classification: cat vs. dog, spam vs. not spam, etc.
- Output layer: **Must use Sigmoid activation** (outputs probabilities)

**Example:**
- True label: $y = 1$ (it's a cat)
- Prediction: $\hat{y} = 0.9$ (90% confident it's a cat)
- Loss: $-\log(0.9) \approx 0.105$ (low loss, good prediction)

- Prediction: $\hat{y} = 0.1$ (10% confident it's a cat, but it actually is)
- Loss: $-\log(0.1) \approx 2.303$ (high loss, wrong prediction)

---

## 4. What is Categorical Cross-Entropy Loss?

**Answer:**
Categorical Cross-Entropy is a loss function used for **multi-class classification** problems (more than two classes).

**Formula:**
$$L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \cdot \log(\hat{y}_{i,c})$$

Where:
- $y_{i,c}$ is 1 if sample $i$ belongs to class $c$, 0 otherwise (one-hot encoding)
- $\hat{y}_{i,c}$ is the predicted probability for sample $i$ belonging to class $c$
- $C$ is the number of classes
- $n$ is the number of samples

**Simplified for single sample:**
If the true class is $c$, the formula simplifies to:
$$L = -\log(\hat{y}_c)$$

**Properties:**
- Requires probabilities that sum to 1
- Only cares about the predicted probability of the correct class
- High probability for correct class = low loss

**Use Case:**
- Multi-class classification: MNIST (10 digits), CIFAR-10 (10 classes), ImageNet (1000 classes)
- Output layer: **Softmax activation** (outputs probabilities that sum to 1)

---

## 5. When should you use MSE vs. Cross-Entropy?

**Answer:**

**Use MSE for:**
- **Regression problems**: Predicting continuous values
  - House prices, temperatures, stock prices, etc.
- **Output layer**: Linear activation (no activation) or ReLU
- **When**: Your target is a real number, not a class

**Use Cross-Entropy for:**
- **Classification problems**: Predicting discrete classes
  - Binary classification: Binary Cross-Entropy
  - Multi-class classification: Categorical Cross-Entropy
- **Output layer**: 
  - Binary: Sigmoid activation
  - Multi-class: Softmax activation (or use CrossEntropyLoss which applies it internally)
- **When**: Your target is a class label (0, 1, 2, etc.)

**Why not MSE for classification?**
- MSE doesn't understand probabilities
- MSE treats all errors equally, doesn't encourage probability outputs
- Cross-entropy provides better gradients for classification tasks

**Rule of thumb:**
- Continuous values → MSE
- Class labels → Cross-Entropy

---

## 6. What is PyTorch's CrossEntropyLoss and how is it different from Categorical Cross-Entropy?

**Answer:**
PyTorch's `CrossEntropyLoss` is a convenient combination of Softmax and Categorical Cross-Entropy that's optimized for multi-class classification.

**What it does:**
1. **Applies Softmax internally**: Converts raw logits to probabilities
2. **Computes Cross-Entropy**: Calculates the loss
3. **Accepts class indices**: Takes class indices (0, 1, 2, ...) instead of one-hot vectors

**Key Differences from Manual Categorical Cross-Entropy:**

| Aspect | Manual Categorical CE | PyTorch CrossEntropyLoss |
|--------|----------------------|-------------------------|
| **Input** | Probabilities (after softmax) | Raw logits (before softmax) |
| **Labels** | One-hot encoded | Class indices |
| **Softmax** | You apply it yourself | Applied internally |
| **Steps** | 2 steps (softmax + CE) | 1 step (combined) |

**Important Notes:**
- **Don't apply softmax** before CrossEntropyLoss (it does it internally)
- **Don't one-hot encode** labels (just use class indices: 0, 1, 2, ...)
- **Output layer**: No activation (raw logits)

**Example:**
```python
# ✅ CORRECT
output = model(x)  # Raw logits, no softmax
loss = criterion(output, target)  # target = [0, 1, 2, ...]

# ❌ WRONG
output = model(x)
output = F.softmax(output, dim=1)  # Don't do this!
loss = criterion(output, target)
```

---

## 7. Why do we square the error in MSE instead of using absolute error?

**Answer:**
We square the error in MSE for several important reasons:

**1. Penalizes Large Errors More:**
- Squaring amplifies large errors: $(10)^2 = 100$ vs. $|10| = 10$
- A prediction off by 10 units gets penalized 100 times more than a prediction off by 1 unit
- Encourages the model to focus on reducing large errors

**2. Differentiable Everywhere:**
- Squared function is smooth and differentiable at all points
- Absolute value $|x|$ is not differentiable at $x = 0$
- Differentiability is crucial for gradient descent

**3. Mathematical Convenience:**
- Squared errors have nice mathematical properties
- Makes optimization easier (convex function)
- Gradient is simple: $\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$

**4. Statistical Interpretation:**
- MSE corresponds to maximizing likelihood under Gaussian noise assumption
- Has connections to maximum likelihood estimation

**Trade-off:**
- MSE is more sensitive to outliers (large errors dominate)
- Mean Absolute Error (MAE) is more robust to outliers but less smooth

---

## 8. What is the relationship between loss functions and activation functions?

**Answer:**
Loss functions and activation functions must be **compatible** and work together:

**1. Output Layer Activation Must Match Loss Function:**

| Problem Type | Loss Function | Output Activation |
|--------------|---------------|-------------------|
| **Regression** | MSE | Linear (no activation) or ReLU |
| **Binary Classification** | Binary Cross-Entropy | Sigmoid |
| **Multi-Class Classification** | CrossEntropyLoss | No activation (raw logits) |
| **Multi-Class Classification** | Categorical CE (manual) | Softmax |

**2. Why This Matters:**

**Binary Classification Example:**
- BCE requires probabilities (0 to 1)
- Sigmoid outputs probabilities (0 to 1)
- **Match!** ✓

**Multi-Class Example:**
- CrossEntropyLoss applies softmax internally
- If you apply softmax yourself, you're applying it twice!
- **Mismatch!** ✗

**3. Common Mistakes:**
- Using MSE with sigmoid output (wrong for classification)
- Applying softmax before CrossEntropyLoss (double softmax)
- Using BCE without sigmoid (outputs not in [0,1] range)

**Key Principle:** The output activation must produce values in the format expected by the loss function.

---

## 9. What are reduction modes in loss functions?

**Answer:**
Reduction modes determine how the loss is aggregated across multiple samples in a batch.

**Three Reduction Modes:**

1. **`reduction='mean'`** (default):
   - Averages the loss over all samples
   - Returns: $\frac{1}{n} \sum_{i=1}^{n} L_i$
   - Most common choice

2. **`reduction='sum'`**:
   - Sums the loss over all samples
   - Returns: $\sum_{i=1}^{n} L_i$
   - Useful for weighted samples or custom computations

3. **`reduction='none'`**:
   - Returns loss for each sample separately
   - Returns: $[L_1, L_2, \ldots, L_n]$
   - Useful for sample weighting, debugging, or custom loss computations

**When to Use Each:**

- **`mean`**: Default for most cases, easy to interpret, scale-independent
- **`sum`**: When you need total loss (e.g., weighted samples)
- **`none`**: When you need per-sample loss (e.g., applying custom weights, debugging)

**Example:**
- Batch of 3 samples with losses: $[0.1, 0.2, 0.3]$
- `mean`: $(0.1 + 0.2 + 0.3) / 3 = 0.2$
- `sum`: $0.1 + 0.2 + 0.3 = 0.6$
- `none`: $[0.1, 0.2, 0.3]$

---

## 10. What is the gradient of MSE loss?

**Answer:**
The gradient of MSE loss with respect to the prediction $\hat{y}$ is:

$$\frac{\partial L}{\partial \hat{y}} = \frac{2}{n} (\hat{y} - y)$$

Or, more commonly, we absorb the 2 into the learning rate:

$$\frac{\partial L}{\partial \hat{y}} = \frac{1}{n} (\hat{y} - y)$$

**Derivation:**
Starting with $L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$:

$$\frac{\partial L}{\partial \hat{y}_i} = \frac{\partial}{\partial \hat{y}_i} \left[\frac{1}{n} \sum_{j=1}^{n} (y_j - \hat{y}_j)^2\right]$$

$$= \frac{1}{n} \cdot 2(y_i - \hat{y}_i) \cdot (-1)$$

$$= \frac{2}{n} (\hat{y}_i - y_i)$$

**Key Properties:**
- **Linear in error**: Gradient is proportional to the prediction error
- **Simple**: Easy to compute and understand
- **Smooth**: No discontinuities

**Why This Matters:**
- This gradient flows backward through the network during backpropagation
- Large errors → large gradients → large weight updates
- Small errors → small gradients → fine-tuning

---

## 11. What is the gradient of Binary Cross-Entropy loss?

**Answer:**
The gradient of Binary Cross-Entropy loss with respect to the prediction $\hat{y}$ is:

$$\frac{\partial L}{\partial \hat{y}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$$

**Derivation:**
Starting with $L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$:

$$\frac{\partial L}{\partial \hat{y}} = -\left[\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}\right]$$

$$= -\frac{y(1-\hat{y}) - (1-y)\hat{y}}{\hat{y}(1-\hat{y})}$$

$$= -\frac{y - y\hat{y} - \hat{y} + y\hat{y}}{\hat{y}(1-\hat{y})}$$

$$= \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

**Key Properties:**
- **Non-linear**: Gradient depends on the prediction itself
- **Large when wrong**: When $\hat{y}$ is close to 0 or 1 and wrong, gradient is large
- **Small when correct**: When $\hat{y}$ is close to the true label, gradient is small

**Why This is Better Than MSE for Classification:**
- Provides stronger gradients when predictions are wrong
- Encourages confident predictions
- Better suited for probability-based outputs

---

## 12. Why is cross-entropy better than MSE for classification?

**Answer:**
Cross-entropy is better than MSE for classification for several reasons:

**1. Understands Probabilities:**
- Cross-entropy works with probability distributions
- MSE treats predictions as arbitrary numbers, not probabilities
- Classification outputs should be probabilities

**2. Better Gradients:**
- Cross-entropy gradient: $\frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$ (large when wrong)
- MSE gradient: $2(\hat{y} - y)$ (linear, constant)
- Cross-entropy provides stronger learning signal when predictions are wrong

**3. Encourages Confident Predictions:**
- Cross-entropy penalizes uncertain predictions more
- MSE doesn't distinguish between probability 0.5 and 0.9 for a correct prediction
- Cross-entropy rewards confident correct predictions

**4. Statistical Foundation:**
- Cross-entropy corresponds to maximum likelihood estimation for classification
- MSE corresponds to maximum likelihood for regression (Gaussian noise)
- Using the right loss matches the problem's statistical assumptions

**Example:**
- True label: 1 (it's a cat)
- Prediction 1: $\hat{y} = 0.9$ (90% confident)
- Prediction 2: $\hat{y} = 0.5$ (50% confident, uncertain)

**MSE:**
- Prediction 1: $(1 - 0.9)^2 = 0.01$
- Prediction 2: $(1 - 0.5)^2 = 0.25$
- Both are "correct" but MSE doesn't reward confidence

**Cross-Entropy:**
- Prediction 1: $-\log(0.9) \approx 0.105$
- Prediction 2: $-\log(0.5) \approx 0.693$
- Cross-entropy rewards confident correct predictions!

---

## 13. What happens if you use MSE for binary classification?

**Answer:**
Using MSE for binary classification leads to poor performance and slow learning:

**Problems:**

1. **Doesn't Understand Probabilities:**
   - MSE treats outputs as arbitrary numbers
   - Doesn't encourage probability-like behavior
   - Model may output values outside [0, 1] range

2. **Poor Gradients:**
   - MSE gradient: $2(\hat{y} - y)$ (linear, constant)
   - Cross-entropy gradient: $\frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$ (non-linear, stronger when wrong)
   - MSE provides weaker learning signal

3. **Slow Convergence:**
   - Model takes longer to learn
   - May get stuck in suboptimal solutions
   - Requires more training epochs

4. **Doesn't Reward Confidence:**
   - Prediction of 0.6 and 0.9 for a true label of 1 have similar MSE
   - Doesn't encourage the model to be confident in correct predictions

**Example:**
- True label: 1
- Prediction: 0.9
- MSE: $(1 - 0.9)^2 = 0.01$
- BCE: $-\log(0.9) \approx 0.105$

- True label: 1
- Prediction: 0.1 (very wrong!)
- MSE: $(1 - 0.1)^2 = 0.81$
- BCE: $-\log(0.1) \approx 2.303$

While both show the second prediction is worse, cross-entropy provides a much stronger signal (2.303 vs 0.81), leading to faster learning.

**Solution:** Always use Binary Cross-Entropy (or CrossEntropyLoss) for classification problems.

---

## 14. What is the Softmax function and why is it used with cross-entropy?

**Answer:**
Softmax is an activation function that converts raw scores (logits) into a probability distribution over multiple classes.

**Formula:**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

Where $C$ is the number of classes.

**Properties:**
1. **Outputs probabilities**: Each output is between 0 and 1
2. **Sums to 1**: All outputs sum to 1 (valid probability distribution)
3. **Preserves ordering**: Larger logits → larger probabilities
4. **Amplifies differences**: Makes the largest value even more dominant

**Example:**
- Logits: $[2.0, 1.0, 0.1]$
- Softmax: $[0.659, 0.242, 0.099]$
- Class 0 has 65.9% probability, Class 1 has 24.2%, Class 2 has 9.9%
- Sum: $0.659 + 0.242 + 0.099 = 1.0$ ✓

**Why Used with Cross-Entropy:**

1. **Cross-Entropy Needs Probabilities:**
   - Categorical cross-entropy: $L = -\sum y_i \log(\hat{y}_i)$
   - Requires $\hat{y}_i$ to be probabilities (between 0 and 1, sum to 1)
   - Softmax provides exactly this

2. **Mathematical Compatibility:**
   - Softmax + Cross-Entropy have nice gradient properties
   - Gradient simplifies to: $\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$
   - Makes optimization efficient

3. **Interpretability:**
   - Softmax outputs are interpretable as class probabilities
   - Easy to understand: "65% chance of class 0"

**Note:** When using PyTorch's `CrossEntropyLoss`, you don't need to apply softmax yourself—it's done internally!

---

## 15. What are common mistakes when using loss functions?

**Answer:**
Common mistakes when using loss functions:

**1. Using MSE for Classification:**
- **Mistake**: Using MSE loss for classification problems
- **Problem**: MSE doesn't understand probabilities, leads to poor performance
- **Solution**: Use Binary Cross-Entropy or CrossEntropyLoss

**2. Applying Softmax Before CrossEntropyLoss:**
- **Mistake**: Applying softmax, then using CrossEntropyLoss
- **Problem**: Double softmax! CrossEntropyLoss applies it internally
- **Solution**: Don't apply softmax, use raw logits

**3. Forgetting Sigmoid with BCE:**
- **Mistake**: Using Binary Cross-Entropy without sigmoid in output layer
- **Problem**: Outputs may not be in [0, 1] range (not probabilities)
- **Solution**: Always use sigmoid with BCE

**4. One-Hot Encoding with CrossEntropyLoss:**
- **Mistake**: One-hot encoding labels when using CrossEntropyLoss
- **Problem**: CrossEntropyLoss expects class indices (0, 1, 2, ...)
- **Solution**: Use class indices, not one-hot vectors

**5. Wrong Output Activation:**
- **Mistake**: Using ReLU for classification output, or sigmoid for regression
- **Problem**: Output format doesn't match loss function requirements
- **Solution**: Match activation to problem type

**6. Not Understanding Reduction Modes:**
- **Mistake**: Using wrong reduction mode for the task
- **Problem**: Loss scale may be wrong, affecting learning rate
- **Solution**: Use 'mean' for most cases, 'none' for custom weighting

**Summary:**
- Match loss function to problem type (regression vs. classification)
- Match output activation to loss function
- Understand what each loss function expects (probabilities, logits, etc.)

---

## 16. How do loss functions relate to the training process?

**Answer:**
Loss functions are central to the training process:

**1. Forward Pass:**
- Model makes predictions: $\hat{y} = \text{model}(x)$
- Loss function computes error: $L = \text{loss}(\hat{y}, y)$
- This single number represents how wrong the model is

**2. Backward Pass (Backpropagation):**
- Compute gradient of loss: $\frac{\partial L}{\partial W}$
- Gradient flows backward through the network
- Each weight gets updated based on its contribution to the loss

**3. Weight Updates:**
- Optimizer uses gradients to update weights
- Goal: Minimize the loss function
- Process repeats until loss is minimized

**The Training Loop:**
```
1. Forward pass: predictions = model(inputs)
2. Compute loss: loss = loss_function(predictions, targets)
3. Backward pass: loss.backward() (computes gradients)
4. Update weights: optimizer.step() (uses gradients)
5. Repeat
```

**Key Relationships:**
- **Lower loss** = Better predictions
- **Loss gradient** = Direction to move weights
- **Loss shape** = Determines optimization difficulty

**Why Loss Function Choice Matters:**
- Different loss functions create different optimization landscapes
- Some are easier to optimize (smooth, good gradients)
- Some are harder (flat regions, poor gradients)
- Choosing the right loss function is crucial for successful training

---

## 17. What is the difference between loss, cost, and objective function?

**Answer:**
These terms are often used interchangeably, but there are subtle differences:

**Loss Function:**
- Measures error for a **single sample**
- Example: $L_i = (y_i - \hat{y}_i)^2$ for one sample

**Cost Function:**
- Measures error for the **entire dataset** (average or sum of losses)
- Example: $C = \frac{1}{n} \sum_{i=1}^{n} L_i$ (MSE over all samples)
- Often used synonymously with loss function

**Objective Function:**
- The function we want to **optimize** (minimize or maximize)
- Can include the cost function plus regularization terms
- Example: $J = C + \lambda \sum w^2$ (cost + L2 regularization)

**In Practice:**
- Most people use "loss function" for everything
- PyTorch uses "loss" (e.g., `nn.MSELoss()`)
- The distinction is mostly academic

**Key Point:**
- All three refer to measuring prediction error
- The goal is to minimize them during training
- They guide the learning process

---

## 18. Can you use different loss functions for the same problem?

**Answer:**
Yes, but it's usually not recommended unless you have a specific reason:

**When It Might Make Sense:**

1. **Robustness to Outliers:**
   - MSE is sensitive to outliers
   - Mean Absolute Error (MAE) is more robust
   - Use MAE if your data has many outliers

2. **Custom Requirements:**
   - Asymmetric loss (penalize over-prediction more than under-prediction)
   - Domain-specific losses (e.g., medical diagnosis where false negatives are worse)

3. **Multi-Task Learning:**
   - Different tasks in the same model might need different losses
   - Example: Classification + regression in one model

**Why Stick to Standard Losses:**

1. **Proven to Work:**
   - MSE for regression, Cross-Entropy for classification are well-established
   - Extensive research and practice support them

2. **Better Gradients:**
   - Standard losses have good gradient properties
   - Custom losses might have optimization issues

3. **Compatibility:**
   - Standard losses work well with standard optimizers
   - Custom losses might need special handling

**Rule of Thumb:**
- Start with standard losses (MSE for regression, Cross-Entropy for classification)
- Only use custom losses if you have a specific, well-justified reason
- Test thoroughly if using custom losses

---

## 19. What is the relationship between loss and accuracy?

**Answer:**
Loss and accuracy are related but measure different things:

**Loss:**
- **Continuous measure**: How far predictions are from targets
- **Works during training**: Used for optimization
- **More informative**: Tells you not just if you're wrong, but how wrong
- **Example**: Loss of 0.1 vs. 0.5 (both might be "wrong" but different degrees)

**Accuracy:**
- **Discrete measure**: Percentage of correct predictions
- **Works for classification**: Counts correct vs. incorrect
- **Less informative**: Only tells you right/wrong, not how close
- **Example**: 80% accuracy (doesn't tell you about confidence)

**Relationship:**

**For Classification:**
- Lower loss generally → Higher accuracy
- But not always: Model might have low loss but make confident wrong predictions
- Accuracy is what we care about, but loss guides training

**Example:**
- Model A: Loss = 0.1, Accuracy = 95%
- Model B: Loss = 0.2, Accuracy = 90%
- Model A is better on both metrics

**Why We Use Loss, Not Accuracy, for Training:**

1. **Differentiable:**
   - Loss is smooth and differentiable
   - Accuracy is discrete (not differentiable)
   - Can't compute gradients for accuracy

2. **More Informative:**
   - Loss tells you how confident and how wrong
   - Accuracy only tells you right/wrong

3. **Better Optimization:**
   - Loss provides gradient information
   - Accuracy doesn't provide gradients

**Key Insight:**
- We **optimize** loss (minimize it)
- We **evaluate** accuracy (maximize it)
- They're related but serve different purposes

---

## 20. How do you choose a loss function for a new problem?

**Answer:**
Here's a systematic approach to choosing a loss function:

**Step 1: Identify Problem Type**

**Regression (Continuous Output):**
- Predicting real numbers: prices, temperatures, etc.
- **Use: MSE** (Mean Squared Error)
- **Output activation**: Linear (no activation) or ReLU

**Binary Classification (Two Classes):**
- Two options: spam/not spam, cat/dog, etc.
- **Use: Binary Cross-Entropy**
- **Output activation**: Sigmoid

**Multi-Class Classification (Multiple Classes):**
- Multiple options: 10 digits, 1000 object classes, etc.
- **Use: CrossEntropyLoss** (or Categorical Cross-Entropy)
- **Output activation**: No activation (raw logits) with CrossEntropyLoss, or Softmax with manual CE

**Step 2: Consider Special Cases**

**Outliers:**
- If data has many outliers, consider Mean Absolute Error (MAE) instead of MSE

**Imbalanced Classes:**
- Consider weighted cross-entropy or focal loss

**Multi-Task:**
- Different tasks might need different losses combined

**Step 3: Start Simple**

- Begin with standard losses (MSE, Cross-Entropy)
- They work well for most problems
- Only customize if you have a specific need

**Decision Tree:**
```
Is output continuous? 
  → Yes: Use MSE
  → No: Is it binary classification?
    → Yes: Use Binary Cross-Entropy
    → No: Use CrossEntropyLoss (multi-class)
```

**Key Principle:**
Match the loss function to your problem type and output format. The standard choices (MSE for regression, Cross-Entropy for classification) work for 95% of problems.

---

