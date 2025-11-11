## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day7.ipynb](notebooks/day7.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

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