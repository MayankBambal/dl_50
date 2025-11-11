## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day18.ipynb](notebooks/day18.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Combine convolution, pooling, and fully connected layers
- Build a complete CNN architecture
- Train CNN on CIFAR-10 or similar dataset
- Understand CNN architecture patterns
- Learn to visualize CNN feature maps

**Deep Learning Concept(s):**
- CNN architecture: Conv → Pool → Conv → Pool → FC → Output
- Feature hierarchy: edges → patterns → objects
- Flattening: converting 2D feature maps to 1D for FC layers
- Common CNN patterns
- Activation functions in CNNs (ReLU)

**Tools Used:**
- PyTorch: combining `nn.Conv2d()`, `nn.MaxPool2d()`, `nn.Linear()`, `nn.ReLU()`, `nn.Flatten()`
- torchvision: `torchvision.datasets.CIFAR10`
- Functions: `nn.Sequential()`, `nn.Flatten()`, building complete CNN models

**Key Learnings:**
- Building CNN: `nn.Sequential(nn.Conv2d(...), nn.ReLU(), nn.MaxPool2d(...), nn.Flatten(), nn.Linear(...))`
- Flattening: `nn.Flatten()` or `x.view(batch_size, -1)` to reshape before FC layers
- Typical CNN pattern: Conv → ReLU → Pool → (repeat) → Flatten → FC → Output
- Understanding feature map dimensions through the network
- Calculating output sizes at each layer
- Using `model.parameters()` to see total parameters
- Visualizing intermediate feature maps to understand what network learns
- CIFAR-10 dataset: 32x32 color images, 10 classes

**References:**
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 14 (CNNs)
- **Tutorial**: [PyTorch: Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- **Tutorial**: [Building CNNs from Scratch](https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html)
- **Dataset**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---