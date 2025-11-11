## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day5.ipynb](notebooks/day5.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

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