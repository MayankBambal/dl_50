## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day17.ipynb](notebooks/day17.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand the purpose of pooling layers
- Learn max pooling vs. average pooling
- Implement pooling operations
- Understand how pooling reduces spatial dimensions
- Learn when to use different pooling strategies

**Deep Learning Concept(s):**
- Pooling operation: downsampling feature maps
- Max pooling: takes maximum value in window
- Average pooling: takes average value in window
- Downsampling: reducing spatial resolution
- Translation invariance enhancement
- Overfitting reduction
- Pooling kernel size (typically 2x2)

**Tools Used:**
- PyTorch: `torch.nn.MaxPool2d()`, `torch.nn.AvgPool2d()`, `torch.nn.AdaptiveAvgPool2d()`
- Functions: `nn.MaxPool2d(kernel_size=2, stride=2)`, `nn.AvgPool2d(kernel_size=2)`

**Key Learnings:**
- Max pooling: `self.pool = nn.MaxPool2d(kernel_size=2, stride=2)`
- Average pooling: `nn.AvgPool2d(kernel_size=2, stride=2)`
- Pooling reduces spatial dimensions: 32x32 → 16x16 (with 2x2 pool, stride=2)
- Adaptive pooling: `nn.AdaptiveAvgPool2d((1, 1))` forces output to specific size
- Global average pooling: alternative to fully connected layers
- Pooling placement: typically after conv layers
- Understanding that pooling has no learnable parameters
- Combining pooling with convolution for hierarchical feature extraction

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 9.3 (Pooling)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 14.3 (Pooling Layers)
- **Tutorial**: [PyTorch: Pooling Layers](https://pytorch.org/docs/stable/nn.html#pooling-layers)
- **Online**: [Understanding Pooling Layers in CNNs](https://towardsdatascience.com/pooling-layers-in-convolutional-neural-networks-84132e3b5c63)

---