## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day15.ipynb](notebooks/day15.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand the limitations of fully connected networks for image data
- Learn why MLPs don't scale well with image resolution
- Understand parameter explosion problem
- Learn about translation invariance and locality
- Recognize the need for specialized architectures for images

**Deep Learning Concept(s):**
- Parameter explosion in fully connected layers for images
- Spatial relationships in images
- Translation invariance requirement
- Locality: nearby pixels are more related
- Parameter sharing benefits
- Inductive biases for vision tasks

**Tools Used:**
- NumPy/PyTorch for calculating parameter counts
- Functions: Manual calculation of parameters: `input_size * output_size` for MLPs

**Key Learnings:**
- Parameter count calculation: for 28x28 image, MLP with 100 hidden neurons = 784*100 + 100*10 = 78,500 parameters
- For 224x224 image: 50,176 input neurons → massive parameter explosion
- Understanding that MLPs treat each pixel independently
- MLPs don't account for spatial structure
- Introduction to the concept that CNNs will solve these problems
- Understanding why feature extraction should be translation invariant

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 9.1 (Convolution and Pooling - The Convolution Operation)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 14 (CNNs)
- **Video**: [Why Convolutional Neural Networks?](https://www.youtube.com/watch?v=f0t-OCG79-U)
- **Online**: [Understanding CNNs: Why Convolutions?](https://towardsdatascience.com/understanding-convolutional-neural-networks-4e30c52bca40)

---