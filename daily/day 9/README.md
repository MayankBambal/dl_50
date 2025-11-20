## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
  - [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)
  - [Andrew Ng: Bias and Variance](https://www.coursera.org/learn/machine-learning/lecture/4VDH1/bias-and-variance)
- **Blog:**
  - [Day 9: Overfitting — When Your Model Memorizes Instead of Learning](https://medium.com/deep-learning-journal/day-9-overfitting-when-your-model-memorizes-instead-of-learning-b97b5d7fe3ce)
**Level 2 (Junior Data Scientist):**
- **Books:**
  - "Deep Learning" by Ian Goodfellow - Chapter 5.2 (Capacity, Overfitting and Underfitting)
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Blogs:**
  - [Understanding Overfitting](https://www.kaggle.com/dansbecker/underfitting-and-overfitting)

## **Plan**

**Objectives:**
- Understand what overfitting is and why it occurs
- Learn to identify overfitting in training curves
- Understand the bias-variance tradeoff
- Learn how to split data (train/validation/test)
- Visualize overfitting through training and validation metrics

**Deep Learning Concept(s):**
- Overfitting definition and symptoms
- Underfitting definition
- Bias-variance tradeoff
- Training vs. validation vs. test set
- Generalization gap
- Learning curves (training loss vs. validation loss)

**Tools Used:**
- Matplotlib for plotting learning curves
- PyTorch: model evaluation on validation set
- Functions: `matplotlib.pyplot.plot()`, model evaluation metrics

**Key Learnings:**
- Splitting data: `train_size = int(0.8 * len(dataset))`, `train_dataset, val_dataset = random_split(dataset, [train_size, val_size])`
- Plotting training vs validation loss to identify overfitting
- Understanding when validation loss starts increasing while training loss decreases
- Using validation set to monitor model performance
- Understanding the difference between training accuracy and validation accuracy
- Early stopping concept (introduction)

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 5.2 (Capacity, Overfitting and Underfitting)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 4 (Training Models - Overfitting)
- **Tutorial**: [Understanding Overfitting](https://www.kaggle.com/dansbecker/underfitting-and-overfitting)
- **Video**: [Andrew Ng: Bias and Variance](https://www.coursera.org/learn/machine-learning/lecture/4VDH1/bias-and-variance)

---