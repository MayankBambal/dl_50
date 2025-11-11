## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day13.ipynb](notebooks/day13.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand the internal covariate shift problem
- Learn how batch normalization stabilizes training
- Implement batch normalization in PyTorch
- Understand batch normalization during training vs. inference
- Learn when to use batch normalization

**Deep Learning Concept(s):**
- Internal covariate shift
- Batch normalization: normalize activations per mini-batch
- Batch norm formula: BN(x) = γ * ((x - μ) / σ) + β
- Running statistics (mean/variance) for inference
- Batch normalization placement (before or after activation)
- Benefits: faster convergence, higher learning rates, less sensitive to initialization

**Tools Used:**
- PyTorch: `torch.nn.BatchNorm1d()`, `torch.nn.BatchNorm2d()`, `torch.nn.BatchNorm3d()`
- Functions: `nn.BatchNorm1d(num_features)`, understanding `track_running_stats` parameter

**Key Learnings:**
- Adding batch norm: `self.bn = nn.BatchNorm1d(num_features)` or `nn.BatchNorm2d(channels)`
- Batch norm after linear/conv layers: `x = self.bn(self.linear(x))` or `x = self.relu(self.bn(self.conv(x)))`
- Understanding that batch norm has learnable parameters (γ, β)
- Batch norm automatically handles train/eval mode switching
- Understanding running mean and variance computation
- When to use batch norm: deep networks, helps with training stability
- Batch size considerations: batch norm requires meaningful batch statistics
- Layer normalization introduction (for comparison, detailed in later weeks)

**References:**
- **Paper**: Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 8.7.1 (Batch Normalization)
- **Tutorial**: [PyTorch: BatchNorm](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- **Video**: [Batch Normalization Explained](https://www.youtube.com/watch?v=dXB-KQYkzNU)

---