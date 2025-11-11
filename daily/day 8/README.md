## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day8.ipynb](notebooks/day8.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Build your first complete neural network using PyTorch
- Load and preprocess the MNIST dataset
- Train a neural network to classify handwritten digits
- Understand the complete training loop
- Evaluate model performance on test data

**Deep Learning Concept(s):**
- Complete neural network pipeline (data → model → training → evaluation)
- Data loading and preprocessing
- Model definition using `nn.Module`
- Training loop structure
- Validation and testing
- Metrics: accuracy calculation

**Tools Used:**
- PyTorch: `torch.nn`, `torch.optim`, `torch.utils.data`
- torchvision: `torchvision.datasets.MNIST`, `torchvision.transforms`
- Functions: `nn.Linear()`, `nn.Sequential()`, `DataLoader()`, `transforms.ToTensor()`, `transforms.Normalize()`

**Key Learnings:**
- Creating a PyTorch model: `class Net(nn.Module): def __init__(): def forward():`
- Dataset loading: `train_dataset = MNIST(root='data', train=True, transform=transforms)`
- DataLoader: `train_loader = DataLoader(dataset, batch_size=64, shuffle=True)`
- Training loop: iterate over batches, compute loss, backward pass, optimizer step
- Moving tensors to device: `.to(device)` or `.cuda()`
- Model evaluation mode: `model.eval()` and `model.train()`
- Computing accuracy: `(predicted == labels).sum().item() / labels.size(0)`
- Saving and loading models: `torch.save()`, `torch.load()`

**References:**
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 10 (Introduction to Artificial Neural Networks)
- **Tutorial**: [PyTorch: Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- **Tutorial**: [PyTorch: Neural Networks](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- **Dataset**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---