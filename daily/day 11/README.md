## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
  -[Dropout Regularization (C2W1L06)](https://www.youtube.com/watch?v=D8PJAL-MZv8&t=9s)
  -[What is Dropout Regularization | How is it different?](https://www.youtube.com/watch?v=kry2JghtMSY)
- **Blog:**
  -[Day 11: All about Dropout Regularization in Neural Networks](https://medium.com/deep-learning-journal/day-11-all-about-dropout-regularization-in-neural-networks-b975c08bc07e)

**Level 2 (Junior Data Scientist):**
- **Books:**
  - "Deep Learning" by Ian Goodfellow - Chapter 7.12 (Dropout)
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Paper:**
  - Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"

## **Plan**

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