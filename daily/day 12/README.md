## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day12.ipynb](notebooks/day12.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand what hyperparameters are
- Learn important hyperparameters (learning rate, batch size, network architecture)
- Implement manual hyperparameter search
- Understand grid search vs. random search
- Learn to track and compare different hyperparameter configurations

**Deep Learning Concept(s):**
- Hyperparameters vs. parameters (weights/biases)
- Learning rate selection
- Batch size effects (speed vs. stability)
- Network architecture choices (depth, width)
- Hyperparameter search strategies
- Hyperparameter importance and sensitivity

**Tools Used:**
- PyTorch for model training
- Python dictionaries for tracking hyperparameters
- Functions: Manual hyperparameter loops, configuration tracking

**Key Learnings:**
- Learning rate ranges: typically 1e-5 to 1e-1, often use 1e-3 or 3e-4 as starting point
- Batch size selection: powers of 2 (32, 64, 128, 256), GPU memory constraints
- Number of epochs: early stopping based on validation loss
- Network depth and width: balancing capacity vs. overfitting
- Creating hyperparameter configs: `config = {'lr': 0.001, 'batch_size': 64, 'hidden_size': 128}`
- Tracking experiments: logging results for different configurations
- Understanding learning rate schedules (introduction)
- Introduction to automated hyperparameter tuning tools (brief mention)

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 11 (Practical Methodology)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 2 (End-to-End ML Project - Fine-Tuning)
- **Tutorial**: [Hyperparameter Tuning Guide](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
- **Online**: [A Practical Guide to Hyperparameter Tuning](https://towardsdatascience.com/a-practical-guide-to-hyperparameter-tuning-3983c80a5b1a)

---