# Day 13 Interview Questions: Batch Normalization

### 1. What problem does Batch Normalization solve?
**Answer:** Batch Normalization solves the problem of *Internal Covariate Shift*. During training, as the weights of earlier layers are updated, the distribution of their outputs changes. This requires later layers to constantly adapt to new, shifting distributions, slowing down convergence and causing instability. Batch Normalization mitigates this by normalizing the inputs to each layer to maintain a consistent mean and variance.

### 2. Explain the learnable parameters γ (gamma) and β (beta) in Batch Normalization. Why do we need them?
**Answer:** While normalization standardizes the activations to have a zero mean and unit variance, this fixed distribution might limit the network's representational power. γ (scale) and β (shift) are learnable parameters that allow the network to scale and shift the normalized values. If the optimal distribution is not zero-mean and unit variance, the network can adjust them. It even gives the network the ability to completely undo the normalization (by setting γ=σ and β=μ) if the original distribution was optimal.

### 3. How does Batch Normalization behave differently during training vs. inference (evaluation)?
**Answer:** 
- **During Training:** BN normalizes using the mean and variance of the *current mini-batch*. Simultaneously, it updates an exponential moving average (running statistics) of the mean and variance.
- **During Inference:** Finding batch statistics isn't reliable or sometimes even possible (e.g., batch size of 1 during inference). BN leverages the *running statistics* recorded during training to perform deterministic normalization.

### 4. Where is the recommended place to add Batch Normalization in a neural network architecture?
**Answer:** The most widely adopted and recommended placement is: `Linear/Convolutional Layer -> Batch Normalization -> Non-Linear Activation (e.g., ReLU)`. This normalizes the "raw" activations before passing them into the non-linearity, which is consistent with classic architectures such as ResNet. Some research explores placing it after the activation, but before is standard.

### 5. What happens if you use Batch Normalization with a very small batch size (e.g., 1 or 2)?
**Answer:** Batch Normalization relies on the mini-batch to compute mean and variance. With a batch size of 1, the variance is undefined (division by zero risk), and with very small batch sizes, the estimated statistics are highly noisy and unreliable, destabilizing training. In such cases, alternatives like Layer Normalization or Group Normalization should be used.

### 6. Do we still need to initialize weights carefully when using Batch Normalization?
**Answer:** While Batch Normalization makes networks significantly less sensitive to poor initialization (like scaling issues), reasonable initializations are still recommended. BN mitigates problems by self-correcting the scale of outputs, allowing the optimization landscape to be more forgiving, but extremely bad weight initialization can still hinder early training.

### 7. What happens if you forget to set `model.eval()` while executing inference on a model with Batch Normalization layers in PyTorch?
**Answer:** PyTorch's `nn.BatchNorm` layers will keep using the batch statistics of the input data rather than the trained running statistics. This can result in heavily skewed, incorrect predictions, especially if the inference batch size is different from the training batch size or if the inference batch size is 1 (which will crash or output all zeros), because it will normalize the single inference example against itself instead of population stats.
