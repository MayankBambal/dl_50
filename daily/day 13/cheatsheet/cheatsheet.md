# Day 13 Cheatsheet: Batch Normalization

## Core Concepts
- **Internal Covariate Shift**: The problem where later layers in a neural network must continuously adapt to shifting input distributions because earlier layers' weights are changing. This slows down training and causes instability.
- **Batch Normalization (BN)**: A technique that normalizes the inputs to each layer across the mini-batch to have a consistent distribution (mean ~0, variance ~1), stabilizing and accelerating training.

## The Formula
For a mini-batch of activations `x`:
```text
BN(x) = γ * ((x - μ) / σ) + β
```
- **μ**: Mean of the mini-batch
- **σ**: Standard deviation of the mini-batch
- **γ (gamma)**: Learnable scale parameter.
- **β (beta)**: Learnable shift parameter.

> [!NOTE]
> γ and β allow the network to "undo" normalization if needed and learn optimal statistics for the task.

## Training vs. Inference Behavior

| Phase | Statistics Used | Behavior |
| --- | --- | --- |
| **Training** (`model.train()`) | Current mini-batch mean & variance. | Normalizes based on the batch. Updates running statistics. |
| **Inference/Eval** (`model.eval()`) | Running mean & variance (Exponential Moving Average). | Uses consistent learned statistics. Independent of batch size. |

## PyTorch Implementation cheat sheet
```python
# 1D Batch Normalization (after Fully Connected layers)
self.bn1d = nn.BatchNorm1d(num_features) # num_features = output size of linear layer

# 2D Batch Normalization (after Convolutional layers)
self.bn2d = nn.BatchNorm2d(num_channels) # num_channels = output channels of conv layer

# 3D Batch Normalization (for 3D Conv / Video)
self.bn3d = nn.BatchNorm3d(num_channels)
```

## Where to Place BN?
**Standard/Most Common Pattern:** `Linear/Conv -> BatchNorm -> Activation`
```python
x = self.linear(x)
x = self.bn(x)      
x = self.relu(x)    
```

## Benefits of Batch Normalization
1. **Faster Convergence**: Less time spent adapting to shifting inputs.
2. **Higher Learning Rates**: Prevents gradients from exploding/vanishing easily.
3. **Less Sensitive to Initialization**: More forgiving of suboptimal initial weights.
4. **Enables Deeper Networks**: Mitigates vanishing gradients (e.g., ResNet).
5. **Regularization Effect**: Mini-batch noise adds a slight regularization effect (like weak dropout).

## Crucial Considerations
- **Batch Size Matters**: Statistics on very small batches are unreliable. Use `layer norm` or `group norm` if batch size < 8.
- **Train/Eval Modes**: `model.train()` and `model.eval()` are critical to toggle BN's behavior. Failure to do so hurts accuracy.
- **Combining with Dropout**: Often used together, though BN may reduce the need for aggressive dropout. Apply dropout *after* BN and activation.
