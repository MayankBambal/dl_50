# Detailed Course Syllabus

## Overview

This document provides a comprehensive breakdown of each day in the 50 Days of Deep Learning course. For each day, you'll find:

- **Objectives**: What you'll learn and accomplish
- **Deep Learning Concept(s)**: Core concepts covered
- **Tools Used**: Specific libraries, frameworks, and functions you'll work with
- **Key Learnings**: New PyTorch functions, concepts, and techniques
- **References**: Recommended books, tutorials, and papers for deeper learning

**Note**: This course uses PyTorch 2.5+ exclusively as our deep learning framework. We focus solely on Python as our programming language to maximize learning efficiency in this intensive 50-day format.

**Environment**: This course is designed to be used with **Google Colab** throughout. All notebooks are optimized for Google Colab, which provides free GPU access and pre-installed libraries, making it the ideal environment for learning deep learning without local setup hassles.

---

## Week 1: The Absolute Basics (Days 1-7)

### Day 1: Introduction to Deep Learning

**Objectives:**
- Understand the relationship between AI, Machine Learning, and Deep Learning
- Recognize real-world applications of deep learning
- Set up Python environment and install PyTorch
- Understand the basic structure of a neural network conceptually

**Deep Learning Concept(s):**
- Artificial Intelligence hierarchy (AI → ML → DL)
- Supervised vs. Unsupervised vs. Reinforcement Learning
- Deep Learning definition and characteristics
- Introduction to neural networks as universal function approximators

**Tools Used:**
- Google Colab (primary environment for all course work)
- Python 3.12+ (pre-installed in Colab)
- NumPy (for basic array operations)
- PyTorch installation and verification (pre-installed in Colab)
- Jupyter Notebooks (via Google Colab)
- Functions: `torch.__version__`, `numpy.array()`, `import torch`

**Key Learnings:**
- How to verify PyTorch installation: `torch.cuda.is_available()` for GPU support
- Basic tensor concept (though not yet implemented)
- Setting up Google Colab and enabling GPU runtime
- Understanding the difference between deep learning frameworks
- Using Google Colab's free GPU resources for deep learning

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 1 (Introduction)
- **Tutorial**: [PyTorch Getting Started](https://pytorch.org/get-started/locally/)
- **Tutorial**: [Google Colab Introduction](https://colab.research.google.com/notebooks/intro.ipynb)
- **Online**: [Deep Learning Course by fast.ai](https://www.fast.ai/)
- **Video**: Andrew Ng's Deep Learning Specialization - Course 1, Week 1

---

### Day 2: The Perceptron

**Objectives:**
- Understand the mathematical foundation of a single perceptron
- Implement a perceptron from scratch using NumPy
- Learn how a perceptron makes binary classification decisions
- Understand linear separability and the perceptron convergence theorem

**Deep Learning Concept(s):**
- Single-layer perceptron architecture
- Weighted sum: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
- Step function (Heaviside) activation
- Linear decision boundary
- Perceptron learning rule (weight updates)

**Tools Used:**
- NumPy for mathematical operations
- Matplotlib for visualization
- Functions: `numpy.dot()`, `numpy.sum()`, `numpy.array()`, `matplotlib.pyplot.plot()`

**Key Learnings:**
- Manual implementation of forward pass: `output = step_function(np.dot(weights, inputs) + bias)`
- Perceptron weight update rule: `w = w + learning_rate * error * input`
- Understanding that perceptrons can only solve linearly separable problems
- Introduction to the concept of weights and biases

**References:**
- **Book**: "Neural Networks and Deep Learning" by Michael Nielsen - Chapter 1 (The Perceptron)
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 6.1 (Feedforward Networks)
- **Paper**: Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- **Tutorial**: [Scikit-learn Perceptron Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)

---

### Day 3: Multi-Layer Perceptrons (MLPs)

**Objectives:**
- Understand why single perceptrons are limited
- Learn the architecture of multi-layer perceptrons (MLPs)
- Implement a simple MLP from scratch using NumPy
- Understand forward propagation through multiple layers
- Learn how to represent networks mathematically

**Deep Learning Concept(s):**
- Multi-layer architecture (input → hidden → output)
- Universal approximation theorem
- Forward propagation
- Layer-wise computation
- Activation functions in hidden layers (introduction, detailed in Day 5)

**Tools Used:**
- NumPy for matrix operations
- Functions: `numpy.matmul()`, `numpy.dot()`, `numpy.random.randn()` for weight initialization

**Key Learnings:**
- Matrix multiplication for forward pass: `h1 = activation(W1 @ X + b1)`
- Understanding weight matrices shape: `(output_dim, input_dim)`
- Stacking layers: `output = W2 @ activation(W1 @ X + b1) + b2`
- Introduction to the concept of hidden layers and their purpose
- Understanding computational graphs conceptually

**References:**
- **Book**: "Neural Networks and Deep Learning" by Michael Nielsen - Chapter 2 (Sigmoid Neurons) and Chapter 3 (The Architecture of Neural Networks)
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 6 (Deep Feedforward Networks)
- **Paper**: Cybenko, G. (1989). "Approximation by superpositions of a sigmoidal function"
- **Tutorial**: [PyTorch: Neural Networks](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

---

### Day 4: Backpropagation

**Objectives:**
- Understand how neural networks learn from errors
- Learn the mathematical derivation of backpropagation
- Implement backpropagation algorithm from scratch
- Understand gradient computation and chain rule
- Learn how gradients flow backward through the network

**Deep Learning Concept(s):**
- Chain rule in calculus
- Gradient descent algorithm
- Backward pass (reverse pass)
- Gradient of loss with respect to weights
- Gradient of loss with respect to biases
- Weight update using gradients

**Tools Used:**
- NumPy for numerical computations
- Manual gradient computation
- Functions: Manual implementation of `∂L/∂W` and `∂L/∂b`

**Key Learnings:**
- Understanding chain rule: `∂L/∂w = (∂L/∂y) * (∂y/∂z) * (∂z/∂w)`
- Backward propagation: computing gradients layer by layer from output to input
- Weight update: `W = W - learning_rate * ∂L/∂W`
- Understanding why we need the backward pass
- Introduction to computational graphs (conceptually)
- Understanding the difference between forward and backward pass

**References:**
- **Book**: "Neural Networks and Deep Learning" by Michael Nielsen - Chapter 2 (Backpropagation Algorithm)
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 6.5 (Back-Propagation and Other Differentiation Algorithms)
- **Paper**: Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors"
- **Video**: [3Blue1Brown: Backpropagation Calculus](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

---

### Day 5: Activation Functions

**Objectives:**
- Understand why activation functions are necessary
- Learn the properties and use cases of different activation functions
- Implement common activation functions (Sigmoid, Tanh, ReLU)
- Understand the vanishing gradient problem (introduction)
- Choose appropriate activation functions for different layers

**Deep Learning Concept(s):**
- Activation function purpose (introducing non-linearity)
- Sigmoid function: σ(x) = 1/(1 + e^(-x))
- Hyperbolic Tangent (Tanh): tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
- Rectified Linear Unit (ReLU): f(x) = max(0, x)
- Gradient of activation functions
- Vanishing gradient problem (introduction)

**Tools Used:**
- NumPy for mathematical functions
- Matplotlib for plotting activation functions
- PyTorch activation functions: `torch.sigmoid()`, `torch.tanh()`, `torch.relu()`, `torch.nn.ReLU()`
- Functions: `numpy.exp()`, `numpy.maximum()`, `matplotlib.pyplot.plot()`

**Key Learnings:**
- PyTorch activation modules: `nn.Sigmoid()`, `nn.Tanh()`, `nn.ReLU()`, `nn.LeakyReLU()`
- Understanding that ReLU is preferred for hidden layers
- Understanding sigmoid/tanh are useful for output layers in binary classification
- Plotting activation functions to visualize their shapes and gradients
- Understanding derivative of ReLU: 0 for x < 0, 1 for x > 0
- Introduction to dying ReLU problem

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 6.3 (Hidden Units)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 10 (Activation Functions)
- **Paper**: Nair, V., & Hinton, G. E. (2010). "Rectified linear units improve restricted boltzmann machines"
- **Tutorial**: [PyTorch: Activation Functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)

---

### Day 6: Loss Functions

**Objectives:**
- Understand the role of loss functions in training
- Learn when to use different loss functions
- Implement Mean Squared Error (MSE) and Cross-Entropy Loss
- Understand the relationship between loss and model performance
- Learn how loss functions relate to probability distributions

**Deep Learning Concept(s):**
- Loss function definition and purpose
- Mean Squared Error (MSE): L = (1/n) Σ(y_true - y_pred)²
- Binary Cross-Entropy: L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
- Categorical Cross-Entropy: L = -Σ y_i * log(ŷ_i)
- Squared error vs. cross-entropy for classification
- Loss function gradients

**Tools Used:**
- NumPy for manual implementation
- PyTorch loss functions: `torch.nn.MSELoss()`, `torch.nn.BCELoss()`, `torch.nn.CrossEntropyLoss()`
- Functions: `numpy.mean()`, `numpy.sum()`, `numpy.log()`, `torch.log()`

**Key Learnings:**
- PyTorch loss modules: `nn.MSELoss()`, `nn.BCELoss()`, `nn.CrossEntropyLoss()`, `nn.NLLLoss()`
- Using loss functions: `criterion = nn.CrossEntropyLoss()`, `loss = criterion(output, target)`
- Understanding reduction modes: `reduction='mean'`, `reduction='sum'`, `reduction='none'`
- When to use MSE: regression problems
- When to use Cross-Entropy: classification problems
- Understanding that CrossEntropyLoss includes softmax internally
- Manual vs. framework loss computation

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 5.1 (Learning Algorithms) and 6.2.2 (Cost Functions)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 4 (Training Models - Loss Functions)
- **Tutorial**: [PyTorch: Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- **Online**: [Loss Functions Explained](https://towardsdatascience.com/understanding-different-loss-functions-for-neural-networks-dd1ed0274718)

---

### Day 7: Optimizers

**Objectives:**
- Understand how optimizers update model weights
- Learn the gradient descent algorithm (Batch, Stochastic, Mini-batch)
- Implement basic gradient descent from scratch
- Understand learning rate and its importance
- Compare different optimization algorithms (SGD, Momentum, Adam introduction)

**Deep Learning Concept(s):**
- Gradient descent algorithm
- Batch Gradient Descent (BGD)
- Stochastic Gradient Descent (SGD)
- Mini-batch Gradient Descent
- Learning rate (α) hyperparameter
- Momentum (introduction, detailed in Day 14)
- Weight update: w = w - α * ∇w

**Tools Used:**
- NumPy for manual implementation
- PyTorch optimizers: `torch.optim.SGD()`, `torch.optim.Adam()`
- Functions: Manual gradient computation and weight updates

**Key Learnings:**
- PyTorch optimizer initialization: `optimizer = torch.optim.SGD(model.parameters(), lr=0.01)`
- Training loop pattern: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
- Understanding learning rate: too high (divergence), too low (slow convergence)
- SGD parameters: `lr`, `momentum`, `weight_decay`
- Introduction to learning rate schedules
- Understanding the three-step optimizer pattern (zero_grad, backward, step)
- Difference between batch, stochastic, and mini-batch gradient descent

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 8 (Optimization for Training Deep Models)
- **Book**: "Neural Networks and Deep Learning" by Michael Nielsen - Chapter 1 (Gradient Descent)
- **Paper**: Bottou, L. (2010). "Large-scale machine learning with stochastic gradient descent"
- **Tutorial**: [PyTorch: Optimizers](https://pytorch.org/docs/stable/optim.html)
- **Video**: [3Blue1Brown: Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w)

---

## Week 2: Building Your First Practical Model (Days 8-14)

### Day 8: First "Hello World" (MNIST)

**Objectives:**
- Build your first complete neural network using PyTorch
- Load and preprocess the MNIST dataset
- Train a neural network to classify handwritten digits
- Understand the complete training loop
- Evaluate model performance on test data

**Deep Learning Concept(s):**
- Complete neural network pipeline (data → model → training → evaluation)
- Data loading and preprocessing
- Model definition using `nn.Module`
- Training loop structure
- Validation and testing
- Metrics: accuracy calculation

**Tools Used:**
- PyTorch: `torch.nn`, `torch.optim`, `torch.utils.data`
- torchvision: `torchvision.datasets.MNIST`, `torchvision.transforms`
- Functions: `nn.Linear()`, `nn.Sequential()`, `DataLoader()`, `transforms.ToTensor()`, `transforms.Normalize()`

**Key Learnings:**
- Creating a PyTorch model: `class Net(nn.Module): def __init__(): def forward():`
- Dataset loading: `train_dataset = MNIST(root='data', train=True, transform=transforms)`
- DataLoader: `train_loader = DataLoader(dataset, batch_size=64, shuffle=True)`
- Training loop: iterate over batches, compute loss, backward pass, optimizer step
- Moving tensors to device: `.to(device)` or `.cuda()`
- Model evaluation mode: `model.eval()` and `model.train()`
- Computing accuracy: `(predicted == labels).sum().item() / labels.size(0)`
- Saving and loading models: `torch.save()`, `torch.load()`

**References:**
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 10 (Introduction to Artificial Neural Networks)
- **Tutorial**: [PyTorch: Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- **Tutorial**: [PyTorch: Neural Networks](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- **Dataset**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

### Day 9: Overfitting

**Objectives:**
- Understand what overfitting is and why it occurs
- Learn to identify overfitting in training curves
- Understand the bias-variance tradeoff
- Learn how to split data (train/validation/test)
- Visualize overfitting through training and validation metrics

**Deep Learning Concept(s):**
- Overfitting definition and symptoms
- Underfitting definition
- Bias-variance tradeoff
- Training vs. validation vs. test set
- Generalization gap
- Learning curves (training loss vs. validation loss)

**Tools Used:**
- Matplotlib for plotting learning curves
- PyTorch: model evaluation on validation set
- Functions: `matplotlib.pyplot.plot()`, model evaluation metrics

**Key Learnings:**
- Splitting data: `train_size = int(0.8 * len(dataset))`, `train_dataset, val_dataset = random_split(dataset, [train_size, val_size])`
- Plotting training vs validation loss to identify overfitting
- Understanding when validation loss starts increasing while training loss decreases
- Using validation set to monitor model performance
- Understanding the difference between training accuracy and validation accuracy
- Early stopping concept (introduction)

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 5.2 (Capacity, Overfitting and Underfitting)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 4 (Training Models - Overfitting)
- **Tutorial**: [Understanding Overfitting](https://www.kaggle.com/dansbecker/underfitting-and-overfitting)
- **Video**: [Andrew Ng: Bias and Variance](https://www.coursera.org/learn/machine-learning/lecture/4VDH1/bias-and-variance)

---

### Day 10: Regularization (L1, L2)

**Objectives:**
- Understand regularization as a technique to prevent overfitting
- Learn L1 (Lasso) and L2 (Ridge) regularization
- Implement weight decay in PyTorch
- Understand the mathematical differences between L1 and L2
- Visualize the effect of regularization on model weights

**Deep Learning Concept(s):**
- Regularization definition and purpose
- L2 regularization (weight decay): L_total = L_data + λΣw²
- L1 regularization (Lasso): L_total = L_data + λΣ|w|
- Weight decay hyperparameter (λ or α)
- Effect on weight values (L2: shrinks weights, L1: sets some to zero)
- Regularization as a constraint on model capacity

**Tools Used:**
- PyTorch optimizers: `weight_decay` parameter
- NumPy for manual regularization computation
- Functions: `optim.SGD(weight_decay=0.01)`, manual L1/L2 loss computation

**Key Learnings:**
- Adding L2 regularization via optimizer: `optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)`
- Manual regularization: `l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())`
- Understanding that weight_decay in PyTorch is L2 regularization
- Comparing models with and without regularization
- Choosing regularization strength: too high (underfitting), too low (still overfitting)
- L1 regularization for feature selection (sparse models)
- Regularization vs dropout (alternative techniques)

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 7.1 (Regularization)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 4 (Regularization)
- **Tutorial**: [PyTorch: Weight Decay](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD)
- **Online**: [Understanding L1 and L2 Regularization](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)

---

### Day 11: Dropout

**Objectives:**
- Understand dropout as a regularization technique
- Learn how dropout works during training vs. inference
- Implement dropout in PyTorch
- Understand the intuition behind dropout (ensemble of sub-networks)
- Experiment with different dropout rates

**Deep Learning Concept(s):**
- Dropout definition and mechanism
- Random deactivation of neurons during training
- Dropout rate (probability of keeping a neuron active)
- Scaling during inference (multiply by keep probability)
- Dropout as ensemble learning approximation
- Why dropout prevents overfitting

**Tools Used:**
- PyTorch: `torch.nn.Dropout()`, `torch.nn.Dropout2d()`, `torch.nn.Dropout3d()`
- Functions: `nn.Dropout(p=0.5)`, applying dropout in model architecture

**Key Learnings:**
- Adding dropout layer: `self.dropout = nn.Dropout(p=0.5)` in model definition
- Using dropout in forward pass: `x = self.dropout(x)` (training), no dropout during `model.eval()`
- Understanding that dropout is automatically disabled in eval mode
- Dropout rates: typically 0.2-0.5 for hidden layers, less for input layers
- Visualizing the effect of dropout on training curves
- Combining dropout with other regularization techniques
- Understanding dropout spatial variants (Dropout2d for CNNs)

**References:**
- **Paper**: Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 7.12 (Dropout)
- **Tutorial**: [PyTorch: Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
- **Online**: [Understanding Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

---

### Day 12: Hyperparameter Tuning

**Objectives:**
- Understand what hyperparameters are
- Learn important hyperparameters (learning rate, batch size, network architecture)
- Implement manual hyperparameter search
- Understand grid search vs. random search
- Learn to track and compare different hyperparameter configurations

**Deep Learning Concept(s):**
- Hyperparameters vs. parameters (weights/biases)
- Learning rate selection
- Batch size effects (speed vs. stability)
- Network architecture choices (depth, width)
- Hyperparameter search strategies
- Hyperparameter importance and sensitivity

**Tools Used:**
- PyTorch for model training
- Python dictionaries for tracking hyperparameters
- Functions: Manual hyperparameter loops, configuration tracking

**Key Learnings:**
- Learning rate ranges: typically 1e-5 to 1e-1, often use 1e-3 or 3e-4 as starting point
- Batch size selection: powers of 2 (32, 64, 128, 256), GPU memory constraints
- Number of epochs: early stopping based on validation loss
- Network depth and width: balancing capacity vs. overfitting
- Creating hyperparameter configs: `config = {'lr': 0.001, 'batch_size': 64, 'hidden_size': 128}`
- Tracking experiments: logging results for different configurations
- Understanding learning rate schedules (introduction)
- Introduction to automated hyperparameter tuning tools (brief mention)

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 11 (Practical Methodology)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 2 (End-to-End ML Project - Fine-Tuning)
- **Tutorial**: [Hyperparameter Tuning Guide](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
- **Online**: [A Practical Guide to Hyperparameter Tuning](https://towardsdatascience.com/a-practical-guide-to-hyperparameter-tuning-3983c80a5b1a)

---

### Day 13: Batch Normalization

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

### Day 14: Advanced Optimizers (Adam)

**Objectives:**
- Understand limitations of basic SGD
- Learn adaptive learning rate optimizers (Adam, RMSprop)
- Understand momentum and its benefits
- Implement Adam optimizer in PyTorch
- Compare different optimizers on same problem

**Deep Learning Concept(s):**
- Momentum: v_t = βv_{t-1} + (1-β)∇θ (velocity accumulates gradients)
- Adaptive learning rates per parameter
- Adam optimizer: combines momentum + RMSprop
- RMSprop: adaptive learning rate based on recent gradient magnitudes
- Adam formula: combines first moment (mean) and second moment (variance) of gradients
- Learning rate decay strategies

**Tools Used:**
- PyTorch optimizers: `torch.optim.Adam()`, `torch.optim.RMSprop()`, `torch.optim.AdamW()`
- Functions: `optim.Adam(lr=0.001, betas=(0.9, 0.999))`, `optim.RMSprop(lr=0.001)`

**Key Learnings:**
- Adam optimizer: `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`
- Adam hyperparameters: `betas=(0.9, 0.999)` (momentum decay, squared gradient decay), `eps=1e-8`
- AdamW: weight decay fix for Adam (`torch.optim.AdamW`)
- RMSprop: `optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)`
- Comparing optimizers: SGD vs. SGD with momentum vs. Adam
- When to use Adam: default choice for most problems, adaptive learning rates
- When to use SGD: sometimes better for generalization, more control
- Learning rate scheduling: `torch.optim.lr_scheduler.StepLR`, `ReduceLROnPlateau`

**References:**
- **Paper**: Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 8.5 (Algorithms with Adaptive Learning Rates)
- **Tutorial**: [PyTorch: Optimizers](https://pytorch.org/docs/stable/optim.html)
- **Online**: [Adam Optimizer Explained](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)

---

## Week 3: Deep Learning for Images (Days 15-21)

### Day 15: Why MLPs Fail for Images

**Objectives:**
- Understand the limitations of fully connected networks for image data
- Learn why MLPs don't scale well with image resolution
- Understand parameter explosion problem
- Learn about translation invariance and locality
- Recognize the need for specialized architectures for images

**Deep Learning Concept(s):**
- Parameter explosion in fully connected layers for images
- Spatial relationships in images
- Translation invariance requirement
- Locality: nearby pixels are more related
- Parameter sharing benefits
- Inductive biases for vision tasks

**Tools Used:**
- NumPy/PyTorch for calculating parameter counts
- Functions: Manual calculation of parameters: `input_size * output_size` for MLPs

**Key Learnings:**
- Parameter count calculation: for 28x28 image, MLP with 100 hidden neurons = 784*100 + 100*10 = 78,500 parameters
- For 224x224 image: 50,176 input neurons → massive parameter explosion
- Understanding that MLPs treat each pixel independently
- MLPs don't account for spatial structure
- Introduction to the concept that CNNs will solve these problems
- Understanding why feature extraction should be translation invariant

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 9.1 (Convolution and Pooling - The Convolution Operation)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 14 (CNNs)
- **Video**: [Why Convolutional Neural Networks?](https://www.youtube.com/watch?v=f0t-OCG79-U)
- **Online**: [Understanding CNNs: Why Convolutions?](https://towardsdatascience.com/understanding-convolutional-neural-networks-4e30c52bca40)

---

### Day 16: Convolution Layers

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

### Day 17: Pooling Layers

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

### Day 18: Building Your First CNN

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

### Day 19: LeNet-5 Architecture

**Objectives:**
- Study the historical LeNet-5 architecture
- Understand early CNN design principles
- Implement LeNet-5 from scratch
- Learn about architectural evolution
- Understand the impact of LeNet-5 on computer vision

**Deep Learning Concept(s):**
- LeNet-5 architecture (1998)
- Historical significance: first successful CNN
- Architecture: 2 conv layers + 3 FC layers
- Handwritten digit recognition
- Tanh activations (historical, now use ReLU)
- Subsampling layers (historical pooling)

**Tools Used:**
- PyTorch: implementing LeNet-5 architecture
- Functions: Building LeNet-5 with specific layer configurations

**Key Learnings:**
- LeNet-5 architecture details: Conv(6) → Pool → Conv(16) → Pool → FC(120) → FC(84) → FC(10)
- Understanding historical vs. modern CNN design
- Appreciating architectural evolution
- Implementing specific architectures from papers
- Understanding that modern CNNs build on these foundations
- Learning to read and implement architectures from research papers

**References:**
- **Paper**: LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 9.9 (Convolutional Networks - Historical Context)
- **Online**: [LeNet-5 Architecture Explained](https://medium.com/@mayurmalviya/lenet-5-a-classic-cnn-architecture-2735f58c8915)
- **Tutorial**: [Implementing LeNet-5 in PyTorch](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

---

### Day 20: Transfer Learning

**Objectives:**
- Understand what transfer learning is and why it's powerful
- Learn to use pre-trained models
- Fine-tune pre-trained models for new tasks
- Understand feature extraction vs. fine-tuning
- Load pre-trained models from torchvision

**Deep Learning Concept(s):**
- Transfer learning definition
- Pre-trained models and their benefits
- Feature extraction: freeze backbone, train classifier
- Fine-tuning: update all or some layers
- Domain adaptation
- Why ImageNet pre-training works

**Tools Used:**
- torchvision models: `torchvision.models.resnet18()`, `torchvision.models.vgg16()`, `torchvision.models.efficientnet_b0()`
- Functions: `models.resnet18(pretrained=True)`, freezing layers: `param.requires_grad = False`, modifying classifier head

**Key Learnings:**
- Loading pre-trained model: `model = torchvision.models.resnet18(pretrained=True)` or `weights='IMAGENET1K_V1'`
- Freezing layers: `for param in model.parameters(): param.requires_grad = False`
- Modifying classifier: `model.fc = nn.Linear(model.fc.in_features, num_classes)` for ResNet
- Two transfer learning approaches:
  - Feature extraction: freeze backbone, only train classifier
  - Fine-tuning: train all layers (or last few) with lower learning rate
- Learning rate scheduling for fine-tuning: typically lower LR for pre-trained parts
- Understanding that pre-trained models learned general features
- Using different architectures: ResNet, VGG, EfficientNet, etc.

**References:**
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 14.6 (Transfer Learning)
- **Tutorial**: [PyTorch: Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- **Tutorial**: [torchvision Models](https://pytorch.org/vision/stable/models.html)
- **Online**: [Transfer Learning Guide](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751)

---

### Day 21: Practical Project (VGG/ResNet)

**Objectives:**
- Apply transfer learning to a real problem
- Use VGG or ResNet architecture
- Train on custom dataset
- Implement data augmentation
- Compare fine-tuning vs. feature extraction

**Deep Learning Concept(s):**
- VGG architecture: deep network with small filters (3x3)
- ResNet architecture: residual connections, skip connections
- Data augmentation for better generalization
- Custom dataset creation
- Evaluation metrics for image classification

**Tools Used:**
- torchvision: `torchvision.models.vgg16()`, `torchvision.models.resnet50()`
- torchvision transforms: `transforms.Compose()`, `transforms.RandomHorizontalFlip()`, `transforms.RandomRotation()`, `transforms.ColorJitter()`
- Custom datasets: `torch.utils.data.Dataset`
- Functions: Building custom dataset classes, applying augmentation

**Key Learnings:**
- VGG architecture: `model = torchvision.models.vgg16(weights='IMAGENET1K_V1')`
- ResNet architecture: `model = torchvision.models.resnet50(weights='IMAGENET1K_V1')`
- Data augmentation: `transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor()])`
- Creating custom dataset: `class CustomDataset(Dataset): def __init__(), def __len__(), def __getitem__()`
- Fine-tuning strategies:
  - Option 1: Train only classifier with frozen backbone
  - Option 2: Fine-tune last few layers
  - Option 3: Fine-tune all layers with discriminative learning rates
- Understanding residual connections: identity mapping + learned residual
- Using different pre-trained models for different tasks
- Evaluating on custom test set

**References:**
- **Paper**: Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG)
- **Paper**: He, K., et al. (2016). "Deep Residual Learning for Image Recognition" (ResNet)
- **Tutorial**: [PyTorch: Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- **Tutorial**: [Custom Datasets in PyTorch](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- **Online**: [Understanding ResNet](https://towardsdatascience.com/understanding-and-visualizing-resnets-442d831b41b8)

---

## Week 4: Deep Learning for Text (Days 22-28)

### Day 22: Sequential Data

**Objectives:**
- Understand what sequential data is and why it's different
- Learn about time series and text data
- Understand variable-length sequences
- Learn about padding and masking
- Prepare text data for neural networks

**Deep Learning Concept(s):**
- Sequential data characteristics
- Time dependencies
- Variable-length sequences
- Padding sequences to fixed length
- Masking padded tokens
- Tokenization basics

**Tools Used:**
- PyTorch: `torch.nn.utils.rnn.pad_sequence()`, `torch.nn.utils.rnn.pack_padded_sequence()`
- Functions: Manual padding, sequence length tracking

**Key Learnings:**
- Padding sequences: `pad_sequence(sequences, batch_first=True, padding_value=0)`
- Packing sequences: `pack_padded_sequence(padded, lengths, batch_first=True)`
- Understanding that sequences have temporal dependencies
- Text preprocessing: tokenization (introduction)
- Sequence length vs. batch dimension
- Masking: ignoring padding tokens in loss computation
- Understanding why we need special architectures for sequences

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.1 (Recurrent Neural Networks - Introduction)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 15 (Processing Sequences Using RNNs and CNNs)
- **Tutorial**: [PyTorch: Working with Variable Length Sequences](https://pytorch.org/tutorials/beginner/seq2seq_translation_tutorial.html)
- **Online**: [Understanding Sequential Data](https://towardsdatascience.com/understanding-rnn-and-lstm-f7cdf6dfc14e)

---

### Day 23: RNNs

**Objectives:**
- Understand the architecture of Recurrent Neural Networks
- Learn how RNNs maintain hidden state
- Implement RNN from scratch
- Understand forward pass through time
- Learn to use PyTorch RNN modules

**Deep Learning Concept(s):**
- RNN architecture: hidden state propagation
- Recurrent connection: h_t = f(W_hh * h_{t-1} + W_xh * x_t + b)
- Unrolling RNN through time
- Sharing weights across time steps
- Hidden state as memory
- Backpropagation through time (BPTT)

**Tools Used:**
- PyTorch: `torch.nn.RNN()`, `torch.nn.RNNCell()`
- Functions: `nn.RNN(input_size, hidden_size, num_layers, batch_first=True)`

**Key Learnings:**
- Creating RNN: `self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)`
- RNN forward pass: `output, hidden = rnn(input, hidden)`
- Understanding hidden state shape: `(num_layers, batch_size, hidden_size)`
- Single vs. multi-layer RNNs
- Understanding that RNN processes sequences step by step
- Output shape: `(batch_size, seq_len, hidden_size)`
- Last hidden state vs. all outputs
- Bidirectional RNNs (introduction)

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.2 (Recurrent Neural Networks)
- **Book**: "Neural Networks and Deep Learning" by Michael Nielsen - Chapter 5 (Deep Learning - Recurrent Neural Networks)
- **Tutorial**: [PyTorch: RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- **Paper**: Elman, J. L. (1990). "Finding structure in time"

---

### Day 24: Vanishing/Exploding Gradients

**Objectives:**
- Understand the vanishing gradient problem in RNNs
- Learn why gradients vanish or explode
- Understand the mathematical causes
- Learn techniques to mitigate these problems (introduction)
- Visualize gradient flow

**Deep Learning Concept(s):**
- Vanishing gradient problem
- Exploding gradient problem
- Backpropagation through time
- Gradient multiplication through many time steps
- Mathematical analysis: gradient = product of many terms
- Why RNNs struggle with long sequences

**Tools Used:**
- PyTorch: gradient clipping
- Functions: `torch.nn.utils.clip_grad_norm_()`, visualizing gradients

**Key Learnings:**
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Understanding why gradients vanish: repeated multiplication of values < 1
- Understanding why gradients explode: repeated multiplication of values > 1
- Effect on learning: network can't learn long-term dependencies
- Gradient clipping for exploding gradients
- Understanding that this problem motivates LSTM/GRU (next days)
- Monitoring gradients during training

**References:**
- **Paper**: Hochreiter, S. (1991). "Untersuchungen zu dynamischen neuronalen Netzen"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.7 (Challenges with Long Sequences)
- **Tutorial**: [Gradient Clipping in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- **Video**: [Vanishing Gradients Explained](https://www.youtube.com/watch?v=qhXZsFVxGKo)

---

### Day 25: LSTMs

**Objectives:**
- Understand LSTM architecture and how it solves vanishing gradients
- Learn about LSTM gates (forget, input, output)
- Implement LSTM in PyTorch
- Understand cell state vs. hidden state
- Learn when to use LSTMs

**Deep Learning Concept(s):**
- LSTM architecture: cell state and hidden state
- Forget gate: decides what to forget
- Input gate: decides what new information to store
- Output gate: decides what parts of cell state to output
- Cell state: long-term memory pathway
- How LSTMs solve vanishing gradient problem

**Tools Used:**
- PyTorch: `torch.nn.LSTM()`, `torch.nn.LSTMCell()`
- Functions: `nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)`

**Key Learnings:**
- Creating LSTM: `self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)`
- LSTM forward: `output, (hidden, cell) = lstm(input, (hidden, cell))`
- Understanding gates: forget gate, input gate, output gate, candidate values
- Cell state shape: `(num_layers, batch_size, hidden_size)`
- LSTM advantages: better at learning long-term dependencies
- When to use LSTM: sequence modeling, time series, NLP
- Understanding that LSTM has more parameters than RNN
- Bidirectional LSTMs: `bidirectional=True`

**References:**
- **Paper**: Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.10 (Long Short-Term Memory)
- **Tutorial**: [PyTorch: LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- **Visual Guide**: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

### Day 26: GRUs

**Objectives:**
- Understand GRU architecture as a simpler alternative to LSTM
- Compare GRU vs. LSTM
- Implement GRU in PyTorch
- Understand reset gate and update gate
- Learn when to prefer GRU over LSTM

**Deep Learning Concept(s):**
- GRU architecture: simpler than LSTM
- Update gate: decides how much past information to keep
- Reset gate: decides how much past information to forget
- Single hidden state (no separate cell state)
- GRU vs. LSTM comparison
- Computational efficiency

**Tools Used:**
- PyTorch: `torch.nn.GRU()`, `torch.nn.GRUCell()`
- Functions: `nn.GRU(input_size, hidden_size, num_layers, batch_first=True)`

**Key Learnings:**
- Creating GRU: `self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)`
- GRU forward: `output, hidden = gru(input, hidden)`
- GRU has 2 gates vs. LSTM's 3 gates (simpler)
- GRU advantages: fewer parameters, faster training, often similar performance
- When to use GRU: when you want simpler model, less data
- When to use LSTM: when you need maximum capacity, very long sequences
- Understanding that GRU is a good default choice
- Bidirectional GRUs: `bidirectional=True`

**References:**
- **Paper**: Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.10 (Gated RNNs)
- **Tutorial**: [PyTorch: GRU Documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- **Online**: [GRU vs. LSTM](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)

---

### Day 27: Bidirectional LSTMs

**Objectives:**
- Understand bidirectional RNNs and their benefits
- Learn how bidirectional LSTMs process sequences
- Implement bidirectional LSTM in PyTorch
- Understand forward and backward passes
- Learn when bidirectional models are useful

**Deep Learning Concept(s):**
- Bidirectional processing: forward and backward passes
- Concatenating forward and backward hidden states
- Context from both past and future
- When bidirectional is beneficial
- Architecture differences

**Tools Used:**
- PyTorch: `torch.nn.LSTM(bidirectional=True)`
- Functions: `nn.LSTM(..., bidirectional=True)`

**Key Learnings:**
- Bidirectional LSTM: `nn.LSTM(..., bidirectional=True)`
- Output size: `hidden_size * 2` (forward + backward concatenated)
- Understanding forward and backward hidden states
- When to use bidirectional: when you have access to full sequence (classification, not generation)
- Not for causal tasks: can't use future info in real-time prediction
- Combining forward and backward: concatenation or other methods
- Use cases: sentiment analysis, named entity recognition, sequence classification
- Understanding that bidirectional doubles parameters

**References:**
- **Paper**: Schuster, M., & Paliwal, K. K. (1997). "Bidirectional recurrent neural networks"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.3 (Bidirectional RNNs)
- **Tutorial**: [Bidirectional LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- **Online**: [Understanding Bidirectional RNNs](https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66)

---

### Day 28: Sentiment Analysis Project

**Objectives:**
- Apply LSTMs to a real NLP task (sentiment analysis)
- Preprocess text data (tokenization, vocabulary)
- Build a complete sentiment analysis model
- Train and evaluate the model
- Understand embedding layers (introduction)

**Deep Learning Concept(s):**
- Sentiment analysis: binary or multi-class text classification
- Text preprocessing pipeline
- Embedding layers: converting words to vectors
- Sequence classification with RNNs
- Using last hidden state for classification

**Tools Used:**
- PyTorch: `torch.nn.Embedding()`, `torch.nn.LSTM()`, `torch.nn.Linear()`
- Text preprocessing: tokenization libraries (basic)
- Functions: `nn.Embedding(vocab_size, embedding_dim)`, building complete NLP model

**Key Learnings:**
- Embedding layer: `self.embedding = nn.Embedding(vocab_size, embedding_dim)`
- Text preprocessing: tokenization, building vocabulary, converting to indices
- Model architecture: Embedding → LSTM → Linear → Output
- Using last hidden state: `output, (hidden, cell) = lstm(...)` then `hidden[-1]` for classification
- Padding sequences for batching
- Building vocabulary from training data
- Handling unknown words (UNK token)
- Understanding that embeddings will be covered in detail next week

**References:**
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 15 (Processing Sequences)
- **Tutorial**: [PyTorch: Sentiment Analysis Tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
- **Tutorial**: [Text Classification with PyTorch](https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html)
- **Dataset**: [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)

---

## Week 5: The Bridge to Transformers (Days 29-35)

### Day 29: Word Embeddings (Word2Vec)

**Objectives:**
- Understand what word embeddings are
- Learn Word2Vec (Skip-gram and CBOW)
- Understand how embeddings capture semantic meaning
- Use pre-trained word embeddings
- Visualize word embeddings

**Deep Learning Concept(s):**
- Word embeddings: dense vector representations
- Word2Vec: predicting context (Skip-gram) or word from context (CBOW)
- Semantic relationships in embedding space
- Vector similarity: cosine similarity
- Pre-trained embeddings (GloVe, Word2Vec)

**Tools Used:**
- gensim: `gensim.models.Word2Vec`, loading pre-trained embeddings
- PyTorch: `torch.nn.Embedding()` with pre-trained weights
- Functions: `Word2Vec()`, loading embeddings, similarity calculations

**Key Learnings:**
- Using pre-trained embeddings: loading Word2Vec/GloVe vectors
- Loading into PyTorch: `nn.Embedding.from_pretrained(weights, freeze=True/False)`
- Understanding embedding dimensions: typically 100, 200, 300
- Semantic relationships: king - man + woman ≈ queen
- Finding similar words using cosine similarity
- Training your own embeddings (optional)
- Understanding that embeddings are learnable parameters
- Freezing vs. fine-tuning embeddings

**References:**
- **Paper**: Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 12.4 (Word Embeddings)
- **Tutorial**: [Word2Vec Tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)
- **Online**: [Understanding Word Embeddings](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)

---

### Day 30: GloVe and fastText

**Objectives:**
- Learn about GloVe embeddings
- Understand fastText for subword embeddings
- Compare different embedding methods
- Use pre-trained GloVe embeddings
- Understand when to use different embeddings

**Deep Learning Concept(s):**
- GloVe: Global Vectors for Word Representation
- fastText: subword-level embeddings (handles OOV words)
- Embedding comparison: Word2Vec vs. GloVe vs. fastText
- Global vs. local context
- Handling out-of-vocabulary words

**Tools Used:**
- fasttext: `fasttext` library
- Loading GloVe embeddings
- Functions: Using different embedding types in models

**Key Learnings:**
- GloVe: based on global co-occurrence statistics
- fastText: handles OOV words by using subword information
- Loading GloVe: converting GloVe format to PyTorch embeddings
- fastText advantages: works with misspellings, rare words
- When to use GloVe: when you want pre-trained, high-quality embeddings
- When to use fastText: when you have OOV problems, multiple languages
- Understanding that modern models use learned embeddings (contextual)
- Transition: from static embeddings to contextual embeddings (BERT/GPT)

**References:**
- **Paper**: Pennington, J., et al. (2014). "GloVe: Global Vectors for Word Representation"
- **Paper**: Bojanowski, P., et al. (2017). "Enriching Word Vectors with Subword Information"
- **Tutorial**: [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
- **Tutorial**: [fastText Tutorial](https://fasttext.cc/docs/en/python-module.html)
- **Online**: [Word Embeddings Comparison](https://towardsdatascience.com/word-embeddings-comparison-glove-vs-word2vec-vs-fasttext-48eac9a61b6e)

---

### Day 31: Encoder-Decoder Architecture

**Objectives:**
- Understand encoder-decoder (Seq2Seq) architecture
- Learn how encoder and decoder work together
- Implement encoder-decoder from scratch
- Understand the bottleneck representation
- Apply to machine translation

**Deep Learning Concept(s):**
- Encoder: converts input sequence to fixed-size representation
- Decoder: generates output sequence from representation
- Bottleneck: fixed-size vector (context vector)
- Sequence-to-sequence mapping
- Teacher forcing during training

**Tools Used:**
- PyTorch: building encoder and decoder with RNNs/LSTMs
- Functions: `nn.LSTM()` for encoder and decoder, managing hidden states

**Key Learnings:**
- Encoder: processes input sequence → final hidden state
- Decoder: takes encoder hidden state → generates output sequence
- Context vector: encoder's final hidden state
- Architecture: Encoder RNN → Context Vector → Decoder RNN
- Teacher forcing: feeding true previous token during training
- Inference: using predicted token as next input (autoregressive)
- Understanding the bottleneck problem (motivates attention, next day)
- Handling variable-length sequences

**References:**
- **Paper**: Sutskever, I., et al. (2014). "Sequence to Sequence Learning with Neural Networks"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.4 (Encoder-Decoder Sequence-to-Sequence Architecture)
- **Tutorial**: [PyTorch: Seq2Seq Translation Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- **Online**: [Understanding Encoder-Decoder Architecture](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)

---

### Day 32: Seq2Seq Limitations

**Objectives:**
- Understand the bottleneck problem in Seq2Seq
- Learn why fixed-size context vector is limiting
- Understand information compression issues
- Recognize long sequence handling problems
- See why attention is needed

**Deep Learning Concept(s):**
- Bottleneck problem: compressing all information into fixed-size vector
- Information loss in long sequences
- Fixed context vector limitations
- Difficulty with long-range dependencies
- Why we need dynamic context

**Tools Used:**
- Analysis and visualization of encoder-decoder limitations
- Comparison of different sequence lengths

**Key Learnings:**
- Understanding that fixed-size context loses information
- Long sequences: hard to compress into single vector
- Each position in output attends to same context (not ideal)
- Motivation for attention: different output positions need different parts of input
- Understanding that attention will solve these problems
- Visualizing information bottleneck
- Preparing for attention mechanism (next day)

**References:**
- **Paper**: Bahdanau, D., et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.4 (Limitations)
- **Online**: [The Bottleneck Problem in Seq2Seq](https://towardsdatascience.com/attention-in-neural-networks-e66920838742)

---

### Day 33: Attention Mechanism

**Objectives:**
- Understand the attention mechanism
- Learn how attention weights work
- Implement attention from scratch
- Understand query, key, and value
- Apply attention to Seq2Seq models

**Deep Learning Concept(s):**
- Attention mechanism: dynamically focusing on relevant parts
- Attention weights: learned importance scores
- Query, Key, Value (Q, K, V) framework
- Attention scores: dot product or additive
- Weighted context vector (not fixed)
- Alignment between input and output

**Tools Used:**
- PyTorch: implementing attention mechanism manually
- Functions: computing attention scores, applying weights, creating context vectors

**Key Learnings:**
- Attention computation: `attention_scores = Q @ K.T / sqrt(d_k)`
- Softmax over attention scores: `attention_weights = softmax(attention_scores)`
- Weighted context: `context = attention_weights @ V`
- Attention in Seq2Seq: decoder attends to all encoder states
- Each decoder step gets different context vector (not fixed!)
- Understanding query (decoder), key (encoder), value (encoder)
- Masked attention: preventing looking at future tokens
- Scaled dot-product attention formula

**References:**
- **Paper**: Bahdanau, D., et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate"
- **Paper**: Luong, M. T., et al. (2015). "Effective Approaches to Attention-based Neural Machine Translation"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 12.4.5 (Attention Mechanisms)
- **Tutorial**: [Implementing Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- **Visual Guide**: [Attention Mechanism Explained](https://lilianweng.github.io/posts/2018-06-24-attention/)

---

### Day 34: Visualizing Attention

**Objectives:**
- Visualize attention weights
- Understand what attention patterns mean
- Create attention heatmaps
- Analyze attention in translation tasks
- Interpret model behavior through attention

**Deep Learning Concept(s):**
- Attention visualization techniques
- Attention heatmaps
- Interpreting attention patterns
- Alignment visualization
- Understanding what model focuses on

**Tools Used:**
- Matplotlib: `matplotlib.pyplot.imshow()` for heatmaps
- Seaborn: `seaborn.heatmap()` for better visualizations
- Functions: plotting attention weights as matrices

**Key Learnings:**
- Creating attention heatmaps: `plt.imshow(attention_weights, cmap='Blues')`
- Understanding attention patterns: diagonal = good alignment
- Interpreting attention: which input words map to which output words
- Attention visualization: rows = decoder positions, columns = encoder positions
- Using attention to debug models
- Understanding attention in different tasks
- Attention as interpretability tool

**References:**
- **Tutorial**: [Visualizing Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- **Online**: [Attention Visualization Tools](https://github.com/jessevig/bertviz)
- **Paper**: Attention visualization examples from Bahdanau et al.

---

### Day 35: Seq2Seq Project

**Objectives:**
- Build a complete Seq2Seq model with attention
- Apply to machine translation or similar task
- Train and evaluate the model
- Compare with and without attention
- Understand practical considerations

**Deep Learning Concept(s):**
- Complete Seq2Seq pipeline with attention
- Machine translation workflow
- Evaluation metrics: BLEU score (introduction)
- Handling variable-length sequences
- Practical training considerations

**Tools Used:**
- PyTorch: complete Seq2Seq implementation
- Data: translation datasets
- Functions: Building end-to-end model with attention

**Key Learnings:**
- Complete architecture: Encoder (LSTM) → Attention → Decoder (LSTM) → Output
- Comparing models: with vs. without attention
- Understanding improvement from attention
- Handling special tokens: SOS, EOS, PAD
- Training considerations: teacher forcing, scheduled sampling
- Evaluation: BLEU score basics
- Preparing for transformers (which use self-attention, not just encoder-decoder attention)

**References:**
- **Paper**: Bahdanau, D., et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate"
- **Tutorial**: [PyTorch: Complete Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- **Dataset**: [WMT Translation Dataset](http://www.statmt.org/wmt/)

---

## Week 6: The Transformer Architecture (Days 36-42)

### Day 36: Introduction to Transformers

**Objectives:**
- Understand why Transformers replaced RNNs
- Learn the high-level Transformer architecture
- Understand self-attention concept
- Learn about the "Attention is All You Need" paper
- Compare Transformers vs. RNNs

**Deep Learning Concept(s):**
- Transformer architecture overview
- Why Transformers: parallelization, long-range dependencies
- Self-attention vs. encoder-decoder attention
- Encoder-decoder structure
- Positional encodings (introduction)
- Multi-head attention (introduction)

**Tools Used:**
- Understanding architecture diagrams
- Preparing for implementation

**Key Learnings:**
- Understanding that Transformers use attention only (no RNNs)
- Parallel processing: all positions processed simultaneously
- Better long-range dependencies than RNNs
- Self-attention: attending to all positions in sequence
- Encoder stack and decoder stack
- Understanding that this architecture enables modern LLMs
- Foundation for BERT, GPT, T5

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 12 (Applications)
- **Visual Guide**: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- **Tutorial**: [Transformer Architecture Explained](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

---

### Day 37: Self-Attention

**Objectives:**
- Understand self-attention mechanism in detail
- Learn Q, K, V computation
- Implement self-attention from scratch
- Understand scaled dot-product attention
- Learn attention masks

**Deep Learning Concept(s):**
- Self-attention: attending to all positions including itself
- Query, Key, Value computation
- Scaled dot-product attention formula
- Attention mask: masking future/padding tokens
- Attention weights and their interpretation

**Tools Used:**
- PyTorch: `torch.nn.functional.scaled_dot_product_attention()` (PyTorch 2.0+)
- Manual implementation: matrix multiplications for Q, K, V

**Key Learnings:**
- Self-attention: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`
- Computing Q, K, V: `Q = XW_q`, `K = XW_k`, `V = XW_v` (from same input)
- Scaled dot-product: division by `sqrt(d_k)` prevents softmax saturation
- Attention mask: `masked_fill(mask == 0, float('-inf'))` before softmax
- Understanding attention matrix: `(seq_len, seq_len)` for each head
- Implementing from scratch vs. using PyTorch function
- Batch processing: handling `(batch, seq_len, d_model)` tensors

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need" - Section 3.2.1
- **Tutorial**: [PyTorch: Scaled Dot-Product Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- **Visual Guide**: [Self-Attention Explained](https://jalammar.github.io/illustrated-transformer/)
- **Video**: [Attention Mechanism in Detail](https://www.youtube.com/watch?v=rBCqOTEfxvg)

---

### Day 38: Multi-Head Attention

**Objectives:**
- Understand why multiple attention heads are used
- Learn how multi-head attention works
- Implement multi-head attention
- Understand head diversity
- Learn concatenation and projection

**Deep Learning Concept(s):**
- Multi-head attention: multiple attention mechanisms in parallel
- Different heads learn different relationships
- Head concatenation and linear projection
- Why multiple heads help
- Attention head specialization

**Tools Used:**
- PyTorch: `torch.nn.MultiheadAttention()`
- Manual implementation: splitting into heads, processing, concatenating

**Key Learnings:**
- Multi-head attention: `MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O`
- Creating MultiheadAttention: `nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)`
- Splitting into heads: `d_model = num_heads * d_k`
- Each head: `(batch, num_heads, seq_len, d_k)` after splitting
- Concatenating heads: `(batch, seq_len, d_model)` then linear projection
- Understanding that different heads attend to different patterns
- Using PyTorch's built-in: `attn_output, attn_weights = multihead_attn(q, k, v)`
- Number of heads: typically 8, 16, or model_dim divisible

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need" - Section 3.2.2
- **Tutorial**: [PyTorch: MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- **Visual Guide**: [Multi-Head Attention](https://jalammar.github.io/illustrated-transformer/)
- **Online**: [Understanding Multi-Head Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)

---

### Day 39: Transformer Encoder

**Objectives:**
- Understand Transformer encoder architecture
- Learn encoder block components
- Implement encoder layer
- Understand feed-forward networks
- Learn layer normalization and residual connections

**Deep Learning Concept(s):**
- Encoder block: Multi-Head Attention → Add & Norm → FFN → Add & Norm
- Feed-forward network: two linear layers with ReLU
- Residual connections (skip connections)
- Layer normalization
- Position-wise FFN
- Stacking encoder layers

**Tools Used:**
- PyTorch: `torch.nn.TransformerEncoder()`, `torch.nn.TransformerEncoderLayer()`
- Manual implementation: building encoder blocks

**Key Learnings:**
- Encoder layer: `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)`
- Feed-forward: `FFN(x) = max(0, xW1 + b1)W2 + b2` (typically expands then contracts)
- Residual connection: `output = LayerNorm(x + Sublayer(x))`
- Layer normalization: `nn.LayerNorm(d_model)` (normalizes across features)
- Building encoder: `encoder = nn.TransformerEncoder(encoder_layer, num_layers)`
- Understanding that each encoder layer refines representations
- Dropout for regularization
- Understanding 6 encoder layers in original Transformer

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need" - Section 3.1
- **Tutorial**: [PyTorch: TransformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
- **Visual Guide**: [Transformer Encoder](https://jalammar.github.io/illustrated-transformer/)
- **Online**: [Transformer Architecture Deep Dive](https://towardsdatascience.com/transformer-architecture-explained-8bb59b4e16e8)

---

### Day 40: Transformer Decoder

**Objectives:**
- Understand Transformer decoder architecture
- Learn masked self-attention
- Implement decoder layer
- Understand encoder-decoder attention
- Learn autoregressive generation

**Deep Learning Concept(s):**
- Decoder block: Masked Multi-Head Attention → Encoder-Decoder Attention → FFN
- Masked self-attention: preventing looking at future tokens
- Encoder-decoder attention: decoder queries attend to encoder outputs
- Autoregressive generation
- Causal masking

**Tools Used:**
- PyTorch: `torch.nn.TransformerDecoder()`, `torch.nn.TransformerDecoderLayer()`
- Functions: implementing causal masks

**Key Learnings:**
- Decoder layer: `nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)`
- Masked self-attention: causal mask (lower triangular)
- Encoder-decoder attention: `decoder_output @ encoder_output.T`
- Building decoder: `decoder = nn.TransformerDecoder(decoder_layer, num_layers)`
- Causal mask: `torch.triu(torch.ones(seq_len, seq_len)) == 0`
- Understanding autoregressive: generating one token at a time
- Decoder uses both: masked self-attention (past tokens) + encoder attention (input)
- Inference: iterative generation with teacher forcing removed

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need" - Section 3.1
- **Tutorial**: [PyTorch: TransformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html)
- **Visual Guide**: [Transformer Decoder](https://jalammar.github.io/illustrated-transformer/)
- **Online**: [Understanding Transformer Decoder](https://towardsdatascience.com/how-does-the-transformer-decoder-work-6b682b1ef7a3)

---

### Day 41: Positional Encodings

**Objectives:**
- Understand why positional encodings are needed
- Learn sinusoidal positional encodings
- Implement positional encoding
- Understand learned vs. fixed encodings
- Learn relative positional encodings (introduction)

**Deep Learning Concept(s):**
- Why positional encodings: attention is permutation-invariant
- Sinusoidal encodings: fixed, mathematical patterns
- Learned positional embeddings
- Positional encoding addition to input embeddings
- Relative vs. absolute positions

**Tools Used:**
- PyTorch: implementing positional encodings
- Functions: creating sinusoidal patterns, learned embeddings

**Key Learnings:**
- Sinusoidal encoding: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`, `PE(pos, 2i+1) = cos(...)`
- Adding to embeddings: `x = embedding + positional_encoding`
- Learned embeddings: `nn.Embedding(max_seq_len, d_model)` (learned)
- Fixed vs. learned: sinusoidal is fixed, embeddings are learned parameters
- Understanding that position info enables understanding of order
- Implementing: creating positional encoding matrix
- Positional encoding shape: `(max_seq_len, d_model)`
- Understanding that modern models often use learned positional embeddings

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need" - Section 3.5
- **Tutorial**: [Positional Encoding Implementation](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- **Online**: [Understanding Positional Encodings](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- **Paper**: Shaw, P., et al. (2018). "Self-Attention with Relative Position Representations" (relative positions)

---

### Day 42: Complete Transformer Review

**Objectives:**
- Review complete Transformer architecture
- Build full Transformer from scratch
- Understand all components working together
- Apply to a simple task
- Compare Transformer vs. previous architectures

**Deep Learning Concept(s):**
- Complete Transformer: Encoder + Decoder + Attention mechanisms
- End-to-end architecture
- Input/output processing
- Putting all pieces together

**Tools Used:**
- PyTorch: `torch.nn.Transformer()`
- Building complete model from scratch
- Functions: integrating all components

**Key Learnings:**
- Complete Transformer: `nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, ...)`
- Input processing: embeddings + positional encoding
- Encoder: processes source sequence
- Decoder: generates target sequence with encoder attention
- Output: linear layer to vocabulary size
- Understanding full architecture end-to-end
- Comparison: Transformer vs. RNN/Seq2Seq (parallelization, performance)
- Preparing for BERT (encoder-only) and GPT (decoder-only)

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need"
- **Tutorial**: [PyTorch: Complete Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- **Visual Guide**: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- **Code**: [Transformer Implementation](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py)

---

## Week 7: The Models That Changed the World (Days 43-50)

### Day 43: BERT Architecture

**Objectives:**
- Understand BERT as encoder-only Transformer
- Learn BERT architecture details
- Understand bidirectional context
- Learn about BERT variants (BERT-base, BERT-large)
- Compare BERT vs. original Transformer

**Deep Learning Concept(s):**
- BERT: Bidirectional Encoder Representations from Transformers
- Encoder-only architecture (no decoder)
- Bidirectional self-attention (can see full sequence)
- Layer stacking (12 or 24 layers)
- Pre-training vs. fine-tuning

**Tools Used:**
- Hugging Face Transformers: `transformers.BertModel`, `transformers.BertConfig`
- Functions: `BertModel.from_pretrained()`, understanding architecture

**Key Learnings:**
- Loading BERT: `from transformers import BertModel; model = BertModel.from_pretrained('bert-base-uncased')`
- BERT architecture: only Transformer encoder (no decoder)
- Bidirectional: unlike GPT, BERT sees full context both ways
- BERT variants: BERT-base (12 layers, 768 hidden), BERT-large (24 layers, 1024 hidden)
- Understanding that BERT is pre-trained, then fine-tuned
- BERT embeddings: contextual (different for same word in different contexts)
- Tokenizer: WordPiece tokenization
- Special tokens: [CLS], [SEP], [MASK], [PAD]

**References:**
- **Paper**: Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- **Tutorial**: [Hugging Face: BERT Documentation](https://huggingface.co/docs/transformers/model_doc/bert)
- **Tutorial**: [Using BERT](https://huggingface.co/docs/transformers/training)
- **Visual Guide**: [BERT Explained](https://jalammar.github.io/illustrated-bert/)

---

### Day 44: Masked Language Modeling

**Objectives:**
- Understand how BERT is pre-trained
- Learn Masked Language Modeling (MLM) objective
- Implement MLM training
- Understand next sentence prediction (NSP)
- Learn pre-training vs. fine-tuning

**Deep Learning Concept(s):**
- Masked Language Modeling: predicting masked tokens
- Pre-training objective: learning language representations
- 15% masking strategy
- Next Sentence Prediction (NSP)
- Self-supervised learning

**Tools Used:**
- Hugging Face: `transformers.BertForMaskedLM`, `transformers.DataCollatorForLanguageModeling`
- Functions: masking tokens, training MLM model

**Key Learnings:**
- MLM: mask 15% of tokens, predict original tokens
- Masking strategy: 80% [MASK], 10% random token, 10% unchanged
- Loading MLM model: `BertForMaskedLM.from_pretrained('bert-base-uncased')`
- Data collator: `DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)`
- Pre-training: learning on large unlabeled text (Wikipedia, BooksCorpus)
- Understanding self-supervised learning (no human labels needed)
- NSP: predicting if sentence B follows sentence A (binary classification)
- Fine-tuning: adapting pre-trained BERT to specific tasks

**References:**
- **Paper**: Devlin, J., et al. (2018). "BERT: Pre-training..." - Section 3.1, 3.3
- **Tutorial**: [Hugging Face: Masked Language Modeling](https://huggingface.co/docs/transformers/tasks/masked_language_modeling)
- **Tutorial**: [Pre-training BERT](https://huggingface.co/docs/transformers/training)
- **Online**: [Understanding BERT Pre-training](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)

---

### Day 45: BERT Fine-tuning

**Objectives:**
- Fine-tune BERT for downstream tasks
- Apply BERT to text classification
- Learn task-specific fine-tuning strategies
- Understand learning rate scheduling for fine-tuning
- Evaluate fine-tuned BERT

**Deep Learning Concept(s):**
- Fine-tuning: adapting pre-trained model to specific task
- Task-specific heads (classification, QA, NER)
- Transfer learning with BERT
- Discriminative fine-tuning
- Task adaptation

**Tools Used:**
- Hugging Face: `transformers.BertForSequenceClassification`, `transformers.Trainer`, `transformers.AutoModelForSequenceClassification`
- Datasets: GLUE, sentiment analysis datasets
- Functions: `Trainer()`, fine-tuning workflow

**Key Learnings:**
- Sequence classification: `BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)`
- Fine-tuning: train all layers or freeze some, typically use lower learning rate (1e-5 to 5e-5)
- Using Trainer: `Trainer(model=model, args=training_args, train_dataset=dataset, ...)`
- Task-specific heads: add classification head on top of BERT
- Learning rate: much lower than training from scratch (pre-trained weights are good)
- Epochs: typically 3-5 epochs (fewer than training from scratch)
- Evaluation: fine-tuned BERT achieves SOTA on many NLP tasks
- Understanding that fine-tuning adapts general language knowledge to specific task

**References:**
- **Paper**: Devlin, J., et al. (2018). "BERT: Pre-training..." - Section 4
- **Tutorial**: [Hugging Face: Fine-tuning BERT](https://huggingface.co/docs/transformers/training)
- **Tutorial**: [Fine-tuning Tutorial](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- **Dataset**: [GLUE Benchmark](https://gluebenchmark.com/)

---

### Day 46: GPT Architecture

**Objectives:**
- Understand GPT as decoder-only Transformer
- Learn GPT architecture details
- Understand causal (masked) self-attention
- Learn about GPT variants (GPT-1, GPT-2, GPT-3)
- Compare GPT vs. BERT

**Deep Learning Concept(s):**
- GPT: Generative Pre-trained Transformer
- Decoder-only architecture (no encoder)
- Causal/autoregressive self-attention
- Language modeling objective
- Generation capabilities

**Tools Used:**
- Hugging Face: `transformers.GPT2Model`, `transformers.GPT2LMHeadModel`
- Functions: `GPT2LMHeadModel.from_pretrained()`, understanding architecture

**Key Learnings:**
- Loading GPT: `GPT2LMHeadModel.from_pretrained('gpt2')`
- GPT architecture: only Transformer decoder (masked self-attention, no encoder-decoder attention)
- Causal masking: can only attend to previous tokens (autoregressive)
- GPT vs. BERT: GPT is generative (predicts next token), BERT is bidirectional
- GPT variants: GPT-1 (117M), GPT-2 (1.5B), GPT-3 (175B parameters)
- Understanding that GPT generates text token by token
- Training: predicting next token given previous tokens (causal LM)
- Tokenizer: BPE (Byte Pair Encoding)

**References:**
- **Paper**: Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training" (GPT-1)
- **Paper**: Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
- **Tutorial**: [Hugging Face: GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- **Visual Guide**: [GPT Explained](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)

---

### Day 47: Causal Language Modeling

**Objectives:**
- Understand causal language modeling objective
- Learn how GPT generates text
- Implement text generation with GPT
- Understand autoregressive generation
- Learn sampling strategies (greedy, top-k, nucleus)

**Deep Learning Concept(s):**
- Causal language modeling: P(token_t | tokens_<t)
- Autoregressive generation: generating one token at a time
- Generation strategies: greedy, sampling, top-k, nucleus (top-p)
- Temperature for controlling randomness
- Prompting: providing context for generation

**Tools Used:**
- Hugging Face: `transformers.GPT2LMHeadModel`, `transformers.GPT2Tokenizer`
- Generation: `model.generate()` function
- Functions: text generation, sampling strategies

**Key Learnings:**
- Generation: `model.generate(input_ids, max_length=100, num_return_sequences=1)`
- Greedy: `do_sample=False` (always picks highest probability token)
- Sampling: `do_sample=True, temperature=1.0`
- Top-k sampling: `top_k=50` (sample from top k tokens)
- Nucleus sampling: `top_p=0.9` (sample from tokens covering 90% probability mass)
- Autoregressive: each generated token becomes input for next step
- Prompting: providing initial text, model continues from there
- Understanding that GPT learns from next-token prediction during training

**References:**
- **Paper**: Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners"
- **Tutorial**: [Hugging Face: Text Generation](https://huggingface.co/docs/transformers/tasks/language_modeling)
- **Tutorial**: [Text Generation with GPT-2](https://huggingface.co/blog/how-to-generate)
- **Online**: [Understanding Text Generation](https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277)

---

### Day 48: Large Language Models (GPT-3)

**Objectives:**
- Understand the scale of large language models
- Learn about GPT-3 capabilities
- Understand few-shot learning
- Learn about in-context learning
- Understand scaling laws

**Deep Learning Concept(s):**
- Large language models: billions/trillions of parameters
- GPT-3: 175B parameters
- Few-shot learning: providing examples in prompt
- In-context learning: learning from examples without gradient updates
- Scaling: more parameters → better performance
- Emergent abilities: capabilities that appear at scale

**Tools Used:**
- Understanding LLM capabilities
- API usage: OpenAI API (conceptual)
- Functions: understanding prompt engineering

**Key Learnings:**
- Understanding that scale matters: GPT-3 (175B) vs. GPT-2 (1.5B)
- Few-shot learning: providing 1-5 examples in prompt, model generalizes
- In-context learning: model learns task from examples without fine-tuning
- Prompting strategies: zero-shot, one-shot, few-shot
- Understanding that LLMs show emergent abilities (reasoning, code generation)
- API access: using LLMs via APIs (OpenAI, Anthropic, etc.)
- Understanding limitations: hallucinations, bias, computation cost
- Preparing for modern LLMs (GPT-4, Claude, etc.)

**References:**
- **Paper**: Brown, T., et al. (2020). "Language Models are Few-Shot Learners" (GPT-3)
- **Paper**: Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models"
- **Online**: [GPT-3 Explained](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)
- **Tutorial**: [OpenAI API Documentation](https://platform.openai.com/docs)

---

### Day 49: BERT vs. GPT vs. T5

**Objectives:**
- Compare encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) architectures
- Understand when to use each architecture
- Learn about T5 (Text-to-Text Transfer Transformer)
- Understand task-specific model selection
- Learn about modern model families

**Deep Learning Concept(s):**
- Architecture comparison: encoder-only, decoder-only, encoder-decoder
- BERT: bidirectional, good for understanding tasks
- GPT: autoregressive, good for generation tasks
- T5: encoder-decoder, good for both understanding and generation
- Task-specific architectures
- Modern model families

**Tools Used:**
- Hugging Face: comparing different model architectures
- Understanding model selection

**Key Learnings:**
- BERT: best for classification, NER, QA (understanding tasks)
- GPT: best for text generation, completion (generation tasks)
- T5: unified architecture, all tasks as text-to-text (translation, summarization, classification as text)
- Understanding that modern models often use hybrid approaches
- Model families: BERT family (RoBERTa, ALBERT), GPT family (GPT-2, GPT-3, GPT-4), T5 family
- Task selection: choose architecture based on task type
- Understanding that some modern models blur these distinctions (decoder-only for everything)

**References:**
- **Paper**: Raffel, C., et al. (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5)
- **Tutorial**: [Hugging Face: Model Overview](https://huggingface.co/docs/transformers/model_summary)
- **Online**: [BERT vs. GPT vs. T5](https://towardsdatascience.com/bert-vs-gpt-vs-t5-a-comparison-of-the-three-major-models-12f6d7e3c3b5)
- **Online**: [Understanding Different Transformer Architectures](https://huggingface.co/docs/transformers/model_summary)

---

### Day 50: Final Project

**Objectives:**
- Apply all learned concepts to a comprehensive project
- Choose appropriate architecture for your task
- Fine-tune a pre-trained model
- Evaluate and deploy (conceptual)
- Demonstrate understanding of deep learning pipeline

**Deep Learning Concept(s):**
- End-to-end deep learning project
- Model selection and architecture design
- Fine-tuning pre-trained models
- Evaluation and metrics
- Practical considerations

**Tools Used:**
- All tools learned throughout the course
- Hugging Face Transformers
- PyTorch
- Complete project workflow

**Key Learnings:**
- Project planning: defining problem, selecting architecture, preparing data
- Using appropriate pre-trained models (BERT, GPT, or others)
- Fine-tuning strategy: learning rates, epochs, evaluation
- Evaluation: appropriate metrics for your task
- Documentation: explaining your approach and results
- Understanding the complete deep learning lifecycle
- Preparing for production deployment (conceptual)

**References:**
- **Course Review**: All previous days' materials
- **Project Ideas**: Text classification, generation, translation, summarization, QA
- **Best Practices**: [Hugging Face: Model Hub](https://huggingface.co/models)
- **Deployment**: [Hugging Face: Inference API](https://huggingface.co/docs/api-inference/index)

---


