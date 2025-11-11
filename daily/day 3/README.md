## **To Do**

Today's workload is slightly higher than other days but its should cement the idea of a neural network in your mind.

**Level 1 (Product Manager Study):**
- **Video:**
  - [Perceptron Network | Neural Networks](https://www.youtube.com/watch?v=torNuKNLwBE)
  - [But what is a neural network? | Deep learning chapter 1 ](https://www.youtube.com/watch?v=aircAruvnKk&t=955s)

- **Blog:**
  - [Day 3: Multi-Layer Perceptrons (MLPs)](https://medium.com/deep-learning-journal/day-3-multi-layer-perceptrons-mlps-924e1b5ad100)

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day3.ipynb](notebooks/day3.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

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