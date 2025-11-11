## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
    - [Perceptron | Neural Networks ](https://www.youtube.com/watch?v=OFbnpY_k7js)
- **Blog:**
    - [Day 2: The Perceptron](https://medium.com/deep-learning-journal/day-2-the-perceptron-650121815780)

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day2.ipynb](notebooks/day2.ipynb)
- **Books:**
    - "Neural Networks and Deep Learning" by Michael Nielsen - Chapter 1 (The Perceptron)
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)

## **Plan**

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