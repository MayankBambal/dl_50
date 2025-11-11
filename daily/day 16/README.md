## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day16.ipynb](notebooks/day16.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand what convolution operation is
- Learn how convolutional layers work
- Implement convolution from scratch
- Understand filter/kernel concepts
- Learn stride, padding, and output size calculations

**Deep Learning Concept(s):**
- Convolution operation definition
- Filters/kernels and their role
- Feature maps
- Stride: how far the filter moves
- Padding: adding zeros around image
- Output size calculation: (W - F + 2P) / S + 1
- Parameter sharing in convolutions
- Depth: number of filters

**Tools Used:**
- PyTorch: `torch.nn.Conv2d()`, `torch.nn.Conv1d()`, `torch.nn.Conv3d()`
- Functions: `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`, `torch.nn.functional.conv2d()`

**Key Learnings:**
- Creating conv layer: `self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)`
- Understanding channels: RGB images have 3 input channels
- Kernel sizes: typically 3x3, 5x5, 7x7 (odd numbers)
- Stride: `stride=1` (default, moves 1 pixel), `stride=2` (downsamples)
- Padding modes: `padding='same'` (keeps size), `padding='valid'` (no padding), `padding=1` (1 pixel padding)
- Output size calculation with padding: `out_size = (in_size + 2*padding - kernel_size) / stride + 1`
- Parameter count: much fewer than MLPs (shared weights)
- Visualizing feature maps: what filters learn to detect

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 9.2 (Convolution and Pooling)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 14.2 (Convolutional Layers)
- **Tutorial**: [PyTorch: Convolution Layers](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- **Visual Guide**: [A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

---