## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day24.ipynb](notebooks/day24.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

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