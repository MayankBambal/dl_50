## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day14.ipynb](notebooks/day14.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

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