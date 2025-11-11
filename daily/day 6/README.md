## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day6.ipynb](notebooks/day6.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

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