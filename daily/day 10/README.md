## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day10.ipynb](notebooks/day10.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

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