## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
  -[Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
  -[Backpropagation calculus | Deep Learning Chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8)
- **Blog:**
  - [Day 4: Backpropagation — How Networks Learn](https://medium.com/@mayankbambal/day-4-backpropagation-how-networks-learn-1c92f66ea3da)

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day4.ipynb](notebooks/day4.ipynb)
- **Books:**
  - "Neural Networks and Deep Learning" by Michael Nielsen - Chapter 2 (Backpropagation Algorithm)
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**
  - "Deep Learning" by Ian Goodfellow - Chapter 6.5 (Back-Propagation and Other Differentiation Algorithms)

## **Plan**

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