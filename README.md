# 🧠 50 Days of Deep Learning

A comprehensive, day-by-day deep learning course covering everything from neural network basics to transformers and modern LLMs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-orange.svg)](https://pytorch.org/)

## 📚 Course Overview

This 50-day course is designed to take you from deep learning fundamentals to advanced transformer architectures. Each day builds upon previous concepts, with hands-on exercises and practical projects.

### Course Structure

The course is organized into **7 weeks** covering different themes:

- **Week 1**: The Absolute Basics
- **Week 2**: Building Your First Practical Model
- **Week 3**: Deep Learning for Images (CNNs)
- **Week 4**: Deep Learning for Text (RNNs)
- **Week 5**: The Bridge to Transformers (Seq2Seq & Attention)
- **Week 6**: The Transformer Architecture
- **Week 7**: The Models That Changed the World (BERT & GPT)

## 🎯 Learning Objectives

By the end of this course, you will:

- Understand the fundamentals of neural networks and deep learning
- Build and train your first deep learning models
- Implement CNNs for image classification
- Work with RNNs, LSTMs, and GRUs for sequence data
- Understand and implement attention mechanisms
- Build transformer models from scratch
- Fine-tune pre-trained models like BERT and GPT
- Deploy deep learning models in production

## 📋 Prerequisites

- Basic Python programming skills
- Understanding of linear algebra and calculus (helpful but not required)
- Familiarity with NumPy and basic data manipulation
- Enthusiasm to learn! 🚀

## 🚀 Getting Started

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/50-days-deep-learning.git
   cd 50-days-deep-learning
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Jupyter Notebook** (if not already installed)
   ```bash
   pip install jupyter
   jupyter notebook
   ```

For detailed setup instructions, see [docs/installation.md](docs/installation.md).

## 📖 Course Schedule

### Week 1: The Absolute Basics
- [Day 1: What is Deep Learning?](daily/day_01/) - Introduction to Deep Learning
- [Day 2: The Simplest "Brain"](daily/day_02/) - The Perceptron Explained
- [Day 3: From One to Many](daily/day_03/) - Introduction to Neural Networks
- [Day 4: How Do Neural Networks Learn?](daily/day_04/) - A Simple Guide to Backpropagation
- [Day 5: The "On/Off Switch"](daily/day_05/) - Understanding Activation Functions
- [Day 6: Measuring Mistakes](daily/day_06/) - A Beginner's Guide to Loss Functions
- [Day 7: The Learning Engine](daily/day_07/) - Introduction to Optimizers

### Week 2: Building Your First Practical Model
- [Day 8: Your First "Hello World"](daily/day_08/) - MNIST with Keras/PyTorch
- [Day 9: The Big Problem](daily/day_09/) - What is Overfitting?
- [Day 10: Fighting Overfitting (Part 1)](daily/day_10/) - Introduction to Regularization
- [Day 11: Fighting Overfitting (Part 2)](daily/day_11/) - Understanding Dropout
- [Day 12: The Art of Tuning](daily/day_12/) - What Are Hyperparameters?
- [Day 13: Better, Faster Training](daily/day_13/) - The Power of Batch Normalization
- [Day 14: A Better Optimizer](daily/day_14/) - Why Everyone Uses "Adam"

### Week 3: Deep Learning for Images (CNNs)
- [Day 15: Why Can't a Basic Neural Network "See"?](daily/day_15/) - The Limitations of MLPs
- [Day 16: The Building Block of Vision](daily/day_16/) - The Convolution Layer Explained
- [Day 17: Shrinking the Image](daily/day_17/) - Understanding Pooling Layers
- [Day 18: Putting It All Together](daily/day_18/) - Building Your First CNN
- [Day 19: A Look at History](daily/day_19/) - The Architecture of LeNet-5
- [Day 20: The "Shortcut" to Success](daily/day_20/) - What is Transfer Learning?
- [Day 21: Practical Project](daily/day_21/) - Using Pre-trained VGG or ResNet

### Week 4: Deep Learning for Text (RNNs)
- [Day 22: The Challenge of "Sequence"](daily/day_22/) - Introduction to Time Series and Text Data
- [Day 23: Networks with "Memory"](daily/day_23/) - The Recurrent Neural Network (RNN)
- [Day 24: The Problem with RNNs](daily/day_24/) - Vanishing and Exploding Gradients
- [Day 25: The Solution](daily/day_25/) - Long Short-Term Memory (LSTM) Networks
- [Day 26: A Simpler Alternative](daily/day_26/) - The Gated Recurrent Unit (GRU)
- [Day 27: Reading Both Ways](daily/day_27/) - The Power of Bidirectional LSTMs
- [Day 28: Practical Project](daily/day_28/) - Sentiment Analysis with an LSTM

### Week 5: The Bridge to Transformers (Seq2Seq & Attention)
- [Day 29: How to Make Words into Numbers](daily/day_29/) - Introduction to Word Embeddings
- [Day 30: Beyond Word2Vec](daily/day_30/) - Understanding GloVe and fastText
- [Day 31: The Encoder-Decoder Architecture](daily/day_31/) - Building a "Seq2Seq" Model
- [Day 32: The "Bottleneck" Problem](daily/day_32/) - Why Seq2Seq with RNNs Is Limited
- [Day 33: The Big Idea](daily/day_33/) - An Intuitive Guide to the "Attention" Mechanism
- [Day 34: Visualizing Attention](daily/day_34/) - Seeing How a Model Translates a Sentence
- [Day 35: Project](daily/day_35/) - Build a Simple Seq2Seq Model

### Week 6: The Transformer Architecture
- [Day 36: Goodbye RNNs](daily/day_36/) - Introducing the Transformer Architecture
- [Day 37: The "Self-Attention" Mechanism](daily/day_37/) - The Heart of the Transformer
- [Day 38: Why "Multi-Head" Attention?](daily/day_38/) - Looking at the Same Thing in Different Ways
- [Day 39: The Transformer Encoder](daily/day_39/) - What's Inside?
- [Day 40: The Transformer Decoder](daily/day_40/) - How It's Different from the Encoder
- [Day 41: The "Time-Travel" Problem](daily/day_41/) - How Positional Encodings Work
- [Day 42: Review](daily/day_42/) - The Full Transformer, Step-by-Step

### Week 7: The Models That Changed the World (BERT & GPT)
- [Day 43: Meet BERT](daily/day_43/) - The "Encoder-Only" Transformer
- [Day 44: How BERT is Trained](daily/day_44/) - Understanding Masked Language Models
- [Day 45: What is "Fine-Tuning"?](daily/day_45/) - Using BERT for Text Classification
- [Day 46: Meet GPT](daily/day_46/) - The "Decoder-Only" Transformer
- [Day 47: How GPT is Trained](daily/day_47/) - Causal Language Modeling
- [Day 48: The Rise of Large Language Models](daily/day_48/) - What GPT-3 Taught Us
- [Day 49: BERT vs. GPT vs. T5](daily/day_49/) - A Simple Comparison of Modern Architectures
- [Day 50: Final Project & Course Wrap-up](daily/day_50/) - Building Your Own Model

## 📁 Repository Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── LICENSE                   # License file
├── .gitignore               # Git ignore rules
├── plan.csv                 # Course plan with all topics
│
├── daily/                   # Daily lesson folders
│   ├── day_01/
│   │   ├── README.md
│   │   ├── notebooks/
│   │   ├── code/
│   │   ├── data/
│   │   └── outputs/
│   └── ...
│
├── data/                    # Shared datasets
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/                  # Saved models
│   ├── checkpoints/
│   ├── trained/
│   └── pretrained/
│
├── utils/                   # Utility functions
├── scripts/                 # Helper scripts
├── projects/                # Larger projects
├── docs/                    # Documentation
│   ├── syllabus.md
│   ├── installation.md
│   └── references.md
│
└── assets/                  # Images, diagrams, etc.
```

## 🛠️ Technologies Used

- **PyTorch 2.5+** - Primary deep learning framework (latest stable version)
- **Python 3.12+** - Latest Python version
- **NumPy 2.0+** - Numerical computing
- **Pandas 2.2+** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **Jupyter Notebooks** - Interactive learning
- **Transformers** - Hugging Face transformers library (for BERT, GPT, etc.)
- **Scikit-learn** - Machine learning utilities

## 📚 Resources

- [Course Syllabus](docs/syllabus.md)
- [Installation Guide](docs/installation.md)
- [References & Further Reading](docs/references.md)
- [FAQ](docs/faq.md)

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the structure of top deep learning repositories
- Built for learners who want to master deep learning step by step

## 📊 Progress Tracking

Keep track of your progress by checking off completed days:

- [ ] Week 1 (Days 1-7)
- [ ] Week 2 (Days 8-14)
- [ ] Week 3 (Days 15-21)
- [ ] Week 4 (Days 22-28)
- [ ] Week 5 (Days 29-35)
- [ ] Week 6 (Days 36-42)
- [ ] Week 7 (Days 43-50)

---

**Happy Learning! 🚀**

*Start your journey with [Day 1](daily/day_01/) and work your way through all 50 days.*

