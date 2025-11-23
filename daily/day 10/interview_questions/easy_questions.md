# Day 10 - Easy Interview Questions

## 1. What is regularization in deep learning?

**Answer:**
Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function that discourages the model from using large weights. It encourages simpler models that generalize better to unseen data.

**Key Purpose:**
- Prevents overfitting by constraining model complexity
- Encourages smaller weights
- Improves generalization performance

**Mathematical Formulation:**
$$L_{total} = L_{data} + \lambda \cdot R(w)$$

Where:
- $L_{data}$ is the original loss (e.g., cross-entropy, MSE)
- $\lambda$ is the regularization strength (hyperparameter)
- $R(w)$ is the regularization term that penalizes weights

**Common Types:**
- L2 Regularization (Weight Decay): $R(w) = \sum w_i^2$
- L1 Regularization (Lasso): $R(w) = \sum |w_i|$

---

## 2. What is L2 regularization (weight decay)?

**Answer:**
L2 regularization (also called Ridge Regression or Weight Decay) penalizes the **squared magnitude** of weights. It's the most common form of regularization in deep learning.

**Mathematical Formula:**
$$L_{total} = L_{data} + \lambda \sum_{i} w_i^2$$

Where $\lambda$ is the regularization strength (often called `weight_decay` in PyTorch).

**What L2 Does:**
- Shrinks weights toward zero (but rarely sets them to exactly zero)
- Prevents any single weight from becoming too large
- Creates smoother decision boundaries
- Encourages the model to use all features rather than relying on a few

**In PyTorch:**
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    weight_decay=0.0001  # This is lambda (L2 regularization)
)
```

---

## 3. What is L1 regularization (Lasso)?

**Answer:**
L1 regularization (also called Lasso Regression) penalizes the **absolute magnitude** of weights. Unlike L2, it can set some weights to exactly zero, creating sparse models.

**Mathematical Formula:**
$$L_{total} = L_{data} + \lambda \sum_{i} |w_i|$$

Where $\lambda$ is the regularization strength.

**What L1 Does:**
- Sets some weights to exactly zero (sparsity)
- Performs automatic feature selection
- Creates simpler models with fewer non-zero weights
- Useful when you have many irrelevant features

**Key Difference from L2:**
- L1 can eliminate weights completely (sparsity)
- L2 only shrinks weights toward zero

---

## 4. What is the difference between L1 and L2 regularization?

**Answer:**

| Property | L1 (Lasso) | L2 (Ridge/Weight Decay) |
|----------|------------|-------------------------|
| **Penalty** | $\sum \|w_i\|$ | $\sum w_i^2$ |
| **Effect on weights** | Sets some to exactly zero | Shrinks toward zero |
| **Sparsity** | Creates sparse models | Doesn't create sparsity |
| **Feature selection** | Yes (automatic) | No |
| **Gradient** | Constant (sign) | Proportional to weight |
| **Stability** | Less stable | More stable |
| **Use case** | Feature selection, interpretability | General deep learning |
| **Common in DL** | Less common | Very common (default) |

**When to Use:**
- **L2**: Default choice for most deep learning problems
- **L1**: When you need feature selection or sparse models

---

## 5. How do you add L2 regularization in PyTorch?

**Answer:**
The easiest way is to use the `weight_decay` parameter in your optimizer:

```python
import torch.optim as optim

# L2 regularization via weight_decay
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    weight_decay=0.0001  # This is lambda (regularization strength)
)
```

**Important Notes:**
- `weight_decay` in PyTorch is exactly L2 regularization
- Works with all optimizers (SGD, Adam, RMSprop, etc.)
- Typical values: 0.0001 to 0.01
- For Adam, use `AdamW` instead of `Adam` for proper weight decay

**Example with AdamW:**
```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)
```

---

## 6. How do you add L1 regularization in PyTorch?

**Answer:**
L1 regularization must be added manually to the loss function since optimizers only support L2 (via `weight_decay`):

```python
# Compute data loss
loss = criterion(output, target)

# Add L1 regularization manually
l1_reg = sum(p.abs().sum() for p in model.parameters())
l1_lambda = 0.001  # Regularization strength
loss = loss + l1_lambda * l1_reg

# Backward pass
loss.backward()
optimizer.step()
```

**What This Does:**
- Computes the sum of absolute values of all weights
- Adds it to the loss with a scaling factor $\lambda$
- The optimizer will minimize this combined loss

**Note:** You can also exclude biases from regularization if desired:
```python
# Regularize only weights (not biases)
l1_reg = sum(p.abs().sum() 
             for name, p in model.named_parameters() 
             if 'bias' not in name)
```

---

## 7. What is the regularization strength (lambda) and how do you choose it?

**Answer:**
The regularization strength $\lambda$ (called `weight_decay` in PyTorch) is a hyperparameter that controls the trade-off between fitting the training data and keeping weights small.

**The Trade-off:**
- **Small $\lambda$ (e.g., 0.0001):** Model focuses on minimizing training loss (may still overfit)
- **Large $\lambda$ (e.g., 0.01):** Model focuses on keeping weights small (may underfit)
- **Optimal $\lambda$:** Balances both (good generalization)

**How to Choose:**
1. **Start with default:** Try `weight_decay=0.0001` or `0.0005`
2. **Use validation set:** Monitor validation loss for different values
3. **Grid search:** Try values like [0, 0.0001, 0.0005, 0.001, 0.01]
4. **Watch for signs:**
   - Too low: Validation loss much higher than training loss
   - Too high: Both training and validation loss are high (underfitting)

**Typical Range:**
- Most problems: 0.0001 to 0.001
- Very complex models: May need up to 0.01
- Simple models: May need less than 0.0001

---

## 8. Why is regularization also called "weight decay"?

**Answer:**
The name "weight decay" comes from the fact that L2 regularization causes weights to decay (shrink) toward zero over time, even without the data loss gradient.

**Mathematical Explanation:**
When we update weights with L2 regularization:
$$w_{new} = w_{old} - \alpha \cdot \left(\frac{\partial L_{data}}{\partial w} + 2\lambda w_{old}\right)$$

Even if the data gradient is zero, we have:
$$w_{new} = w_{old} - \alpha \cdot 2\lambda w_{old} = w_{old}(1 - 2\alpha\lambda)$$

This multiplicative decay factor $(1 - 2\alpha\lambda)$ causes weights to shrink exponentially toward zero at each step—hence "weight decay."

**Key Insight:**
- The regularization term continuously pushes weights toward zero
- This happens at every optimization step
- The effect is cumulative over training

---

## 9. What happens if you use too much regularization?

**Answer:**
Using too much regularization (large $\lambda$) causes **underfitting**—the model becomes too constrained and cannot learn the underlying patterns in the data.

**Symptoms:**
- **High training loss:** Model can't fit the training data well
- **High validation loss:** Model also performs poorly on validation data
- **Small weights:** All weights are very close to zero
- **Simple model:** Model is too simple to capture patterns

**Example:**
```
Epoch 1:  Train Loss: 0.8,  Val Loss: 0.8   (both high)
Epoch 2:  Train Loss: 0.75, Val Loss: 0.75  (both high)
Epoch 3:  Train Loss: 0.7,  Val Loss: 0.7   (both high, not improving)
```

**Solution:**
- Reduce regularization strength ($\lambda$)
- Try smaller values like 0.0001 or 0.0005
- Monitor both training and validation loss

**The Balance:**
- Too little regularization: Overfitting
- Too much regularization: Underfitting
- Just right: Good generalization

---

## 10. What happens if you use too little regularization?

**Answer:**
Using too little regularization (small $\lambda$ or $\lambda = 0$) allows the model to overfit—it memorizes the training data instead of learning generalizable patterns.

**Symptoms:**
- **Low training loss:** Model fits training data very well
- **High validation loss:** Model performs poorly on new data
- **Large weights:** Some weights become very large
- **Large generalization gap:** Big difference between train and validation performance

**Example:**
```
Epoch 1:  Train Loss: 0.5,  Val Loss: 0.6   (gap growing)
Epoch 2:  Train Loss: 0.3,  Val Loss: 0.5   (gap growing)
Epoch 3:  Train Loss: 0.1,  Val Loss: 0.6   (val loss increasing!)
```

**Solution:**
- Increase regularization strength ($\lambda$)
- Try values like 0.0005, 0.001, or 0.01
- Monitor the generalization gap

---

## 11. Should you regularize biases?

**Answer:**
**Generally, no.** It's common practice to regularize only weights, not biases.

**Why:**
- Biases don't cause the same overfitting problems as weights
- Regularizing biases can hurt model performance
- Biases are typically much smaller than weights

**In PyTorch:**
By default, `weight_decay` regularizes all parameters (including biases). To exclude biases:

```python
# Separate parameters
weight_params = [p for n, p in model.named_parameters() if 'bias' not in n]
bias_params = [p for n, p in model.named_parameters() if 'bias' in n]

# Only regularize weights
optimizer = optim.SGD(
    [
        {'params': weight_params, 'weight_decay': 0.0001},
        {'params': bias_params, 'weight_decay': 0}
    ],
    lr=0.01
)
```

**In Practice:**
- Most practitioners use default (regularize everything)
- Excluding biases is a minor optimization
- Focus on getting the right $\lambda$ first

---

## 12. What is the relationship between regularization and overfitting?

**Answer:**
Regularization is a technique specifically designed to **prevent overfitting** by constraining model complexity.

**The Problem (Overfitting):**
- Model learns training data too well
- Memorizes noise and dataset-specific patterns
- Fails to generalize to new data
- Large gap between training and validation performance

**How Regularization Helps:**
- **Penalizes large weights:** Prevents model from becoming too complex
- **Encourages simpler solutions:** Forces model to use smaller weights
- **Reduces generalization gap:** Improves validation performance
- **Better generalization:** Model learns patterns that work on new data

**The Trade-off:**
- Without regularization: Model may overfit (high variance)
- With too much regularization: Model may underfit (high bias)
- With optimal regularization: Good balance (good generalization)

**Visual Example:**
```
Without Regularization:
Train Loss: 0.05, Val Loss: 0.5  (overfitting!)

With Optimal Regularization:
Train Loss: 0.15, Val Loss: 0.2  (good generalization!)

With Too Much Regularization:
Train Loss: 0.6, Val Loss: 0.6  (underfitting!)
```

---

## 13. Can you use both L1 and L2 regularization together?

**Answer:**
Yes! Combining L1 and L2 regularization is called **Elastic Net**. It gives you benefits of both:
- Sparsity from L1 (some weights become zero)
- Smoothness from L2 (weights are shrunk)

**Mathematical Formula:**
$$L_{total} = L_{data} + \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2$$

**In PyTorch:**
```python
# Compute data loss
loss = criterion(output, target)

# Add both L1 and L2
l1_reg = sum(p.abs().sum() for p in model.parameters())
l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l1_lambda * l1_reg + l2_lambda * l2_reg

# Backward pass
loss.backward()
optimizer.step()
```

**When to Use:**
- You want both sparsity and smoothness
- You have many features and want feature selection (L1) but also stability (L2)
- Less common in deep learning (L2 is usually sufficient)

**Note:** In deep learning, L2 regularization (weight_decay) is the standard. Elastic Net is more common in traditional machine learning.

---

## 14. How does regularization affect the training process?

**Answer:**
Regularization affects training in several ways:

**1. Loss Function:**
- Total loss is higher (includes penalty term)
- Model must balance data fitting and weight size

**2. Weight Updates:**
- Weights are pushed toward zero at each step
- Large weights get larger penalty gradients
- Weights decay over time

**3. Training Speed:**
- May converge slightly slower (constrained optimization)
- But often reaches better solutions faster (less overfitting)

**4. Final Weights:**
- Weights are smaller than without regularization
- More uniform weight distribution
- Less extreme values

**5. Generalization:**
- Better validation performance
- Smaller gap between train and validation loss
- More robust to new data

**Example Training Curves:**
```
Without Regularization:
Epoch 1: Train: 0.5, Val: 0.6
Epoch 5: Train: 0.1, Val: 0.5  (overfitting!)

With Regularization:
Epoch 1: Train: 0.5, Val: 0.6
Epoch 5: Train: 0.2, Val: 0.25  (good generalization!)
```

---

## 15. What is the difference between weight_decay in SGD and Adam?

**Answer:**
The `weight_decay` parameter works differently in SGD vs. Adam:

**SGD with weight_decay:**
- Direct L2 regularization
- Weight update: $w = w - \alpha(\nabla L + 2\lambda w)$
- Works as expected

**Adam with weight_decay:**
- Original Adam implementation has a bug
- Weight decay is applied incorrectly (not true L2 regularization)
- Use **AdamW** instead for proper weight decay

**AdamW (Weight Decay Fix):**
- Correctly implements L2 regularization
- Decouples weight decay from gradient-based updates
- Recommended when using weight decay with adaptive optimizers

**In PyTorch:**
```python
# SGD: weight_decay works correctly
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)

# Adam: weight_decay has issues
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  # Not recommended

# AdamW: Use this instead!
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)  # Recommended
```

**Key Takeaway:** Use `AdamW` instead of `Adam` when you want weight decay with adaptive learning rates.

---

## 16. How do you choose between L1 and L2 regularization?

**Answer:**
Choose based on your goals:

**Use L2 (Weight Decay) when:**
- **Standard deep learning:** Default choice for most problems
- **All features matter:** You don't want to eliminate any features
- **Stable training:** You want smooth optimization
- **General purpose:** Works well for most neural networks

**Use L1 (Lasso) when:**
- **Feature selection:** You want to identify which features matter
- **Interpretability:** You want sparse, interpretable models
- **Model compression:** You want fewer non-zero weights
- **High-dimensional data:** Many irrelevant features to eliminate

**In Practice:**
- **Deep Learning:** Almost always use L2 (weight_decay)
- **Traditional ML:** L1 is more common for feature selection
- **Sparse Models:** Use L1 if you specifically need sparsity

**Rule of Thumb:** Start with L2. Only use L1 if you have a specific need for sparsity or feature selection.

---

## 17. What is the effect of regularization on model weights?

**Answer:**
Regularization affects the distribution and magnitude of weights:

**Without Regularization:**
- Weights can become very large
- Some weights dominate (extreme values)
- Weight distribution is spread out
- Model can memorize training data

**With L2 Regularization:**
- Weights are shrunk toward zero
- More uniform weight distribution
- Fewer extreme values
- Weights cluster closer to zero

**With L1 Regularization:**
- Some weights become exactly zero (sparsity)
- Remaining weights are smaller
- Creates sparse weight matrix
- Automatic feature selection

**Visual Comparison:**
```
Weight Distribution:

Without Reg:  [ -5.2,  3.8, -2.1,  4.5, -1.9,  6.2, ...]  (spread out)
With L2:      [ -0.8,  0.6, -0.3,  0.7, -0.2,  0.9, ...]  (clustered near zero)
With L1:      [  0.0,  0.5,  0.0,  0.3,  0.0,  0.7, ...]  (many zeros)
```

**Key Insight:** Regularization creates simpler, more generalizable models by constraining weight values.

---

## 18. How does regularization relate to the bias-variance tradeoff?

**Answer:**
Regularization directly addresses the **variance** part of the bias-variance tradeoff:

**Bias-Variance Tradeoff:**
- **Bias (Underfitting):** Model too simple, can't learn patterns
- **Variance (Overfitting):** Model too complex, memorizes training data
- **Goal:** Balance both for good generalization

**How Regularization Helps:**
- **Reduces Variance:** Prevents overfitting by constraining model complexity
- **May Increase Bias:** Can make model slightly too simple (if over-regularized)
- **Optimal Regularization:** Balances bias and variance

**The Effect:**
```
Without Regularization:
- Low bias, High variance → Overfitting

With Optimal Regularization:
- Balanced bias and variance → Good generalization

With Too Much Regularization:
- High bias, Low variance → Underfitting
```

**Key Principle:** Regularization is a tool to control the bias-variance tradeoff by reducing model complexity (variance) while potentially increasing bias slightly.

---

## 19. Can regularization be applied to different layers with different strengths?

**Answer:**
Yes! You can apply different regularization strengths to different layers or parameter groups.

**Why You Might Want This:**
- Early layers: May need less regularization (learn basic features)
- Later layers: May need more regularization (prevent overfitting)
- Fine-tuning: Pre-trained layers need less regularization

**In PyTorch:**
```python
# Different regularization for different layers
optimizer = optim.SGD(
    [
        {'params': model.layer1.parameters(), 'weight_decay': 0.0001},
        {'params': model.layer2.parameters(), 'weight_decay': 0.001},  # More regularization
        {'params': model.classifier.parameters(), 'weight_decay': 0.0005}
    ],
    lr=0.01
)
```

**Common Patterns:**
- **Transfer Learning:** Lower weight_decay for pre-trained layers
- **Deep Networks:** More regularization in later layers
- **Feature Layers:** Less regularization in early feature extraction layers

**In Practice:**
- Most practitioners use the same regularization for all layers
- Different strengths are an advanced technique
- Start with uniform regularization, then experiment if needed

---

## 20. What other techniques work together with regularization to prevent overfitting?

**Answer:**
Regularization is one of many techniques to prevent overfitting. They often work together:

**Complementary Techniques:**
1. **Dropout:** Randomly disables neurons during training (Day 11)
2. **Early Stopping:** Stop training when validation loss stops improving
3. **Data Augmentation:** Artificially increase dataset size
4. **Batch Normalization:** Normalizes activations (Day 13)
5. **Model Architecture:** Choose appropriate model capacity
6. **Ensemble Methods:** Combine multiple models

**How They Work Together:**
- **Regularization + Dropout:** Both constrain model complexity
- **Regularization + Early Stopping:** Regularization helps, early stopping prevents over-training
- **Regularization + Data Augmentation:** Regularization constrains, augmentation provides more data

**Example:**
```python
# Combining multiple techniques
model = Net()
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001  # L2 regularization
)
# Model also uses dropout (Day 11)
# Training uses early stopping
# Data uses augmentation
```

**Key Insight:** Regularization is part of a toolkit. Combining multiple techniques often works better than using just one.

---

